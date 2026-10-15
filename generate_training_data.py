"""Convert stage-II scene grasps into DGS-Net point-level training annotations."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import yaml

from dgs_net.transform import Transform


def read_scene(scene_dir, frame):
    """Read the source dataset's scene metadata and one RGB-D frame."""
    objects = []
    for line in (scene_dir / 'obj_info.txt').read_text().splitlines()[2:]:
        cols = line.split()
        objects.append((int(cols[0]), np.asarray(cols[4:], dtype=float)))
    labels = []
    for line in (scene_dir / 'grasp_lables.txt').read_text().splitlines()[2:]:
        cols = line.split()
        labels.append(dict(t7d=np.asarray(cols[:7], dtype=float),
                           pre_joints=np.asarray(cols[7:27], dtype=float),
                           exprct_joints=np.asarray(cols[27:47], dtype=float),
                           label=int(cols[47]), type=cols[48],
                           target_id=int(cols[49])))
    visible = None
    for line in (scene_dir / 'visible_mask.txt').read_text().splitlines():
        cols = line.split(maxsplit=1)
        if int(cols[0]) == frame:
            visible = [v.strip() == 'True' for v in cols[1].strip('[]').split(',')]
            break
    extrinsic = None
    for line in (scene_dir / 'cam_extrinsic.txt').read_text().splitlines()[1:]:
        cols = line.split()
        if int(cols[0]) == frame:
            extrinsic = np.asarray(cols[1:], dtype=float)
            break
    if visible is None or extrinsic is None:
        raise ValueError(f'Missing frame metadata: {scene_dir}, frame {frame}')
    depth = cv2.imread(str(scene_dir / 'depth' / f'{frame:04d}.png'), cv2.IMREAD_UNCHANGED)
    seg = cv2.imread(str(scene_dir / 'seg' / f'{frame:04d}.png'))
    if depth is None or seg is None:
        raise FileNotFoundError(f'Missing depth or segmentation image: {scene_dir}, frame {frame}')
    return objects, labels, visible, extrinsic, depth, seg


def filtered_pointcloud(depth, seg, intrinsics, min_area=160000):
    """Match the source foreground bounding box, padding and depth filtering."""
    mask = np.all(seg != 0, axis=2)
    if mask.any():
        yy, xx = np.where(mask)
        x0, x1, y0, y1 = xx.min(), xx.max(), yy.min(), yy.max()
    else:
        x0, x1, y0, y1 = 0, depth.shape[1], 0, depth.shape[0]
    x0, y0 = max(0, x0 - 40), max(0, y0 - 40)
    x1, y1 = min(depth.shape[1], x1 + 40), min(depth.shape[0], y1 + 40)
    while (x1 - x0) * (y1 - y0) < min_area:
        old = (x0, x1, y0, y1)
        if x0 > 10:
            x0 -= 10
        if y0 > 10:
            y0 -= 10
        if x1 < depth.shape[1] - 10:
            x1 += 10
        if y1 < depth.shape[0] - 10:
            y1 += 10
        if old == (x0, x1, y0, y1):
            break
    u, v = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    z = depth[y0:y1, x0:x1].astype(float) / 1000.0
    valid = (z != 0) & (z != 1.999)
    fx, fy, cx, cy = intrinsics
    points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=-1)
    return points[valid], seg[y0:y1, x0:x1, 0][valid]


def generate_frame(data_root, scene_id, frame, cfg, rng=None):
    scene_dir = data_root / 'scenes' / f'scene_{scene_id:04d}'
    objects, labels, visible, extrinsic, depth, seg = read_scene(scene_dir, frame)
    cam = cfg['cam']
    points, seg_labels = filtered_pointcloud(depth, seg,
                                             (cam['fx'], cam['fy'], cam['cx'], cam['cy']))
    if not len(points):
        raise ValueError(f'Empty point cloud: scene {scene_id}, frame {frame}')
    voxel_size = 0.004
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    down, _, point_map = cloud.voxel_down_sample_and_trace(
        voxel_size, cloud.get_min_bound() - voxel_size * 0.5,
        cloud.get_max_bound() + voxel_size * 0.5)
    sampled = np.asarray(down.points)
    indices = np.asarray([idx[0] for idx in point_map if len(idx) > 0])
    count = cfg['dataset']['num_points']
    if rng is None:
        rng = np.random
    if len(sampled) > count:
        choice = rng.choice(len(sampled), count, replace=False)
        sampled, indices = sampled[choice], indices[choice]
    elif len(sampled) < count:
        extra = rng.choice(len(sampled), count - len(sampled), replace=True)
        sampled, indices = np.vstack((sampled, sampled[extra])), np.hstack((indices, indices[extra]))
    seg_labels = seg_labels[indices]
    cls_mask = seg_labels.copy()
    cls_mask[cls_mask != 0] = 1
    grasp_mask = -np.ones(count)
    tree = KDTree(sampled)
    object_map = {obj_id: (i, pose) for i, (obj_id, pose) in enumerate(objects)}
    world_to_cam = Transform.from_list(extrinsic)
    positive, neighbors, selected = [], [], []
    for label in labels:
        if label['label'] == 0 or label['target_id'] not in object_map:
            continue
        object_index, object_pose = object_map[label['target_id']]
        if not visible[object_index]:
            continue
        # Keep the composition order used by train_data_generate_TASE.py.
        world_to_tcp = Transform.from_list(object_pose) * Transform.from_list(label['t7d'])
        cam_to_tcp = world_to_cam * world_to_tcp
        tcp = cam_to_tcp.to_list()[-3:]
        nearest_id = int(tree.query(tcp, k=1)[1])
        same_label_count = int(np.count_nonzero(seg_labels == seg_labels[nearest_id]))
        neighbor_count = min(max(int(same_label_count * 0.1), 3), 20)
        _, candidate_ids = tree.query(tcp, k=min(1 + neighbor_count, count))
        for j, point_id in enumerate(np.atleast_1d(candidate_ids)):
            if point_id in positive:
                continue
            positive.append(int(point_id))
            neighbors.extend(int(x) for x in np.concatenate((np.atleast_1d(candidate_ids)[j + 1:],
                                                               np.atleast_1d(candidate_ids)[:j])))
            item = label.copy()
            item['t7d'] = cam_to_tcp.to_list()
            item['grasp_points_id'] = int(point_id)
            pre = item['pre_joints'].reshape(5, 4).copy()
            pre[:, 2:] = pre[:, 2:].mean(axis=1, keepdims=True)
            final = item['exprct_joints'].reshape(5, 4).copy()
            final[:, 2:] = final[:, 2:].max(axis=1, keepdims=True)
            item['pre_joints'] = pre.reshape(-1).tolist()
            item['exprct_joints'] = final.reshape(-1).tolist()
            selected.append(item)
            break
    grasp_mask[neighbors] = 0
    grasp_mask[positive] = 1
    grasp_mask[seg_labels == 0] = -1
    output = scene_dir
    (output / 'pointcloud').mkdir(parents=True, exist_ok=True)
    (output / 'grasp').mkdir(parents=True, exist_ok=True)
    np.save(output / 'pointcloud' / f'{frame:04d}.npy',
            dict(points=sampled, grasp_points_mask=grasp_mask, cls_mask=cls_mask, seg=seg_labels))
    np.save(output / 'grasp' / f'{frame:04d}.npy', selected)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--scene', type=int, required=True)
    parser.add_argument('--frame-start', type=int, default=0)
    parser.add_argument('--frame-end', type=int, default=256, help='Exclusive end')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional per-frame seed for reproducible point sampling')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    for frame in range(args.frame_start, args.frame_end):
        rng = None
        if args.seed is not None:
            frame_seed = np.random.SeedSequence([args.seed, args.scene, frame]).generate_state(1)[0]
            rng = np.random.RandomState(frame_seed)
        count = generate_frame(args.data_root, args.scene, frame, cfg, rng=rng)
        print(f'scene {args.scene:04d}, frame {frame:04d}: {count} grasps')


if __name__ == '__main__':
    main()
