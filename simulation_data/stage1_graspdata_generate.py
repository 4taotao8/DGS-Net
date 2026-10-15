"""Generate object-level grasp labels from the supplied URDF assets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random

import numpy as np
import yaml

from .grasp_methods import (
    bowl_grasp_method_stage1,
    cuboid_grasp_method_stage1,
    cylinder_grasp_method_stage1,
    sphere_grasp_method_stage1,
    stick_grasp_method_stage1,
)
from .saver import DATA_ROOT, save_grasp_lables
from .simulation import Stage1_single_Sim


OBJECT_SETS = ('cuboid', 'cylinder', 'stick', 'sphere', 'cup', 'bowl')
DEFAULT_OBJECT_SETS = ('cuboid', 'cylinder', 'stick', 'sphere', 'bowl')
METHODS = {
    'cuboid': cuboid_grasp_method_stage1,
    'cylinder': cylinder_grasp_method_stage1,
    'stick': stick_grasp_method_stage1,
    'sphere': sphere_grasp_method_stage1,
    'cup': cylinder_grasp_method_stage1,
    'bowl': bowl_grasp_method_stage1,
}


def generate_object(object_set, object_index, scales, attempts, gui, seed):
    """Preserve the source grasp sampler and save each scale's labels."""
    random.seed(seed)
    np.random.seed(seed)
    sim = Stage1_single_Sim(object_set, gui=gui,
                            urdf_root=Path(DATA_ROOT) / 'urdfs')
    counts = []
    try:
        count = sim.discover_objects()
        if not 0 <= object_index < count:
            raise IndexError(f'{object_set} object index {object_index} outside 0..{count - 1}')
        method = METHODS[object_set]
        for scale in scales:
            sim.reset()
            _, name, _, pose, shape_size, _ = sim.gengerate_stage1_scence(object_index, scale)
            sim.save_state()
            labels = []
            successes = 0
            for _ in range(attempts):
                grasp, result = method(sim, pose, shape_size,
                                       sim.gripper.finger_depth,
                                       sim.gripper.max_hand_width)
                labels.append(grasp)
                successes += int(result == 1)
            save_grasp_lables(object_set, name, scale, labels, successes)
            label_file = (Path(DATA_ROOT) / 'grasp_labels' / object_set /
                          Path(name).stem / f'{Path(name).stem}_{scale}.txt')
            if not label_file.is_file():
                raise RuntimeError(f'Failed to write object grasp labels: {label_file}')
            counts.append((str(name), scale, successes))
    finally:
        sim.world.close()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-set', choices=OBJECT_SETS)
    parser.add_argument('--object-index', type=int,
                        help='Generate one object; requires --object-set')
    parser.add_argument('--attempts', type=int,
                        help='Grasp attempts per object and scale')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=6282)
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    if args.object_index is not None and args.object_set is None:
        parser.error('--object-index requires --object-set')
    if args.workers < 1:
        parser.error('--workers must be positive')
    cfg = yaml.safe_load((Path(__file__).with_name('config.yaml')).read_text())
    attempts = cfg['num_grasps_per_scale'] if args.attempts is None else args.attempts
    if attempts < 1:
        parser.error('--attempts must be positive')
    if args.gui and args.workers != 1:
        parser.error('--gui requires --workers 1')
    data_root = Path(DATA_ROOT)
    if not (data_root / 'urdfs/hitdlr3/hand_right_simplify.urdf').is_file():
        parser.error(f'Missing hand URDF under {data_root / "urdfs/hitdlr3"}')
    jobs = []
    for object_set in ([args.object_set] if args.object_set else DEFAULT_OBJECT_SETS):
        root = data_root / 'urdfs' / object_set
        if not root.is_dir():
            parser.error(f'Missing object models: {root}')
        count = len(list(root.glob('*.urdf')))
        indexes = [args.object_index] if args.object_index is not None else range(count)
        for index in indexes:
            jobs.append((object_set, index, cfg['urdf_scale'], attempts, args.gui,
                         args.seed + len(jobs)))
    if args.workers == 1:
        results = map(lambda job: generate_object(*job), jobs)
        for counts in results:
            for name, scale, success in counts:
                print(f'{name}, scale={scale}: {success}/{attempts} successful grasps')
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for counts in pool.map(_run_job, jobs):
                for name, scale, success in counts:
                    print(f'{name}, scale={scale}: {success}/{attempts} successful grasps')


def _run_job(job):
    return generate_object(*job)


if __name__ == '__main__':
    main()
