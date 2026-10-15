"""Validate object grasps in multi-object scenes and render RGB-D views."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

import numpy as np
import yaml

from .grasp_methods import stage2_grasp
from .saver import (
    DATA_ROOT,
    find_grasp_lables_file,
    read_grasp_labels_stage1,
    save_image_pointcloud,
    save_scene,
    save_stage2_grasp_lables,
    save_visible_mask,
)
from .simulation import Stage2_multi_Sim


OBJECT_SETS = ('cuboid', 'cylinder', 'stick', 'bowl', 'sphere')


def scene_accepted(successes, minimum, shuffle):
    if not successes:
        return False
    failures = sum(n < minimum for n in successes)
    count = len(successes)
    if shuffle:
        allowed = (0 if count < 3 else 1 if count < 5 else
                   2 if count < 7 else 3 if count < 9 else
                   4 if count < 11 else 5 if count <= 13 else 7)
    else:
        allowed = 0 if count <= 5 else 1 if count <= 7 else 2 if count <= 12 else 3
    return failures <= allowed


def object_visible(ids, segmentation):
    return [bool(np.sum(segmentation == obj_id) >= 500) for obj_id in ids]


def generate_scene(scene_id, cfg, shuffle=False, max_attempts=100,
                   object_count=(5, 15), views=(8, 32), max_stage1_grasps=None):
    rng = np.random.default_rng(seed=scene_id + 7816)
    root = Path(DATA_ROOT)
    sim = Stage2_multi_Sim(gui=cfg.get('sim_gui', False), data_root=str(root),
                           urdf_root=root / 'urdfs', random=rng)
    try:
        if sim.discover_objects(OBJECT_SETS) < object_count[0]:
            raise ValueError('Not enough object URDFs for the requested scene size.')
        for attempt in range(1, max_attempts + 1):
            sim.reset()
            number = int(rng.integers(object_count[0], object_count[1] + 1))
            ids, names, classes, scales, poses = sim.gengerate_stage2_scence(
                number, list(cfg['urdf_scale']), cfg['min_grasps_per_urdf'],
                shuffle=shuffle)
            if len(ids) < object_count[0]:
                continue
            filenames = find_grasp_lables_file(str(root), names, classes, scales)
            successes = []
            labels = []
            for filename, pose, obj_id in zip(filenames, poses, ids):
                _, _, _, _, object_labels = read_grasp_labels_stage1(filename)
                if max_stage1_grasps is not None:
                    object_labels = object_labels[:max_stage1_grasps]
                success, checked_labels, _ = stage2_grasp(
                    sim, object_labels, pose, obj_id)
                successes.append(success)
                labels.extend(checked_labels)
            if not scene_accepted(successes, cfg['min_grasps_per_scene'], shuffle):
                continue

            output = root / 'scenes' / f'scene_{scene_id:04d}'
            if output.exists():
                shutil.rmtree(output)
            save_stage2_grasp_lables(scene_id, labels, successes)
            save_scene(scene_id, ids, names, classes, scales, poses, sim)
            frame = 0
            for theta in np.linspace(np.pi * 3 / 24, np.pi * 9 / 24, views[0]):
                for phi in np.linspace(0, 2 * np.pi, views[1]):
                    rgb, depth, seg, extrinsic = sim.render_image(1, theta, phi)
                    save_visible_mask(scene_id, frame, object_visible(ids, seg))
                    save_image_pointcloud(scene_id, frame, rgb, depth, seg,
                                          extrinsic, sim)
                    frame += 1
            required = (output / 'grasp_lables.txt', output / 'obj_info.txt',
                        output / 'depth' / f'{frame - 1:04d}.png',
                        output / 'seg' / f'{frame - 1:04d}.png')
            if not all(path.is_file() for path in required):
                raise RuntimeError(f'Incomplete generated scene: {output}')
            return scene_id, len(ids), sum(successes), attempt
        raise RuntimeError(f'Scene {scene_id} did not pass validation after {max_attempts} attempts.')
    finally:
        sim.world.close()


def _run_job(job):
    return generate_scene(*job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=int, help='Generate one scene ID')
    parser.add_argument('--start-scene', type=int)
    parser.add_argument('--end-scene', type=int, help='Exclusive end')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-attempts', type=int, default=100)
    parser.add_argument('--shuffle-count', type=int)
    parser.add_argument('--seed', type=int, default=6282)
    parser.add_argument('--min-stage1-success', type=int)
    parser.add_argument('--min-scene-success', type=int)
    parser.add_argument('--objects-min', type=int, default=5)
    parser.add_argument('--objects-max', type=int, default=15)
    parser.add_argument('--views-elevation', type=int)
    parser.add_argument('--views-azimuth', type=int)
    parser.add_argument('--max-stage1-grasps', type=int,
                        help='Cap labels per object for a quick simulation check')
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    if args.min_stage1_success is not None:
        cfg['min_grasps_per_urdf'] = args.min_stage1_success
    if args.min_scene_success is not None:
        cfg['min_grasps_per_scene'] = args.min_scene_success
    cfg['sim_gui'] = args.gui
    start = cfg['start_scene_id'] if args.start_scene is None else args.start_scene
    end = cfg['end_scene_id'] if args.end_scene is None else args.end_scene
    scene_ids = [args.scene] if args.scene is not None else list(range(start, end))
    shuffle_count = cfg['num_shuffle_scene'] if args.shuffle_count is None else args.shuffle_count
    if (args.workers < 1 or args.max_attempts < 1 or args.objects_min < 1 or
            args.objects_max < args.objects_min or shuffle_count < 0 or
            shuffle_count > end - start):
        parser.error('Invalid workers, scene size, attempt count, or shuffle count.')
    if args.gui and args.workers != 1:
        parser.error('--gui requires --workers 1')
    views = (cfg['views_elevation'] if args.views_elevation is None else args.views_elevation,
             cfg['views_azimuth'] if args.views_azimuth is None else args.views_azimuth)
    if min(views) < 1:
        parser.error('View counts must be positive.')
    if args.max_stage1_grasps is not None and args.max_stage1_grasps < 1:
        parser.error('--max-stage1-grasps must be positive.')
    candidates = np.arange(start, end)
    shuffled = set(np.random.default_rng(args.seed).choice(
        candidates, size=shuffle_count, replace=False).tolist())
    jobs = [(scene_id, cfg, scene_id in shuffled, args.max_attempts,
             (args.objects_min, args.objects_max), views, args.max_stage1_grasps)
            for scene_id in scene_ids]
    if args.workers == 1:
        results = map(_run_job, jobs)
        for scene_id, objects, grasps, attempts in results:
            print(f'scene {scene_id:04d}: {objects} objects, {grasps} validated grasps '
                  f'after {attempts} attempts')
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for scene_id, objects, grasps, attempts in pool.map(_run_job, jobs):
                print(f'scene {scene_id:04d}: {objects} objects, {grasps} validated grasps '
                      f'after {attempts} attempts')


if __name__ == '__main__':
    main()
