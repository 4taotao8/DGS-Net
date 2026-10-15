"""Predict DGS-Net grasps from an RGB image and a millimeter depth image."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import yaml

from dgs_net.loss import compute_rotation_matrix_from_ortho6d
from dgs_net.model import PointTransformerV3, dataloader_preprosess


def rgb_depth_to_pointcloud(depth, rgb, fx, fy, cx, cy):
    """Back-project depth in millimeters into camera-frame points in meters."""
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth.astype(np.float32) / 1000.0
    return np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=-1).reshape(-1, 3), \
        rgb.reshape(-1, 3).astype(np.float32) / 255.0


def decode_predictions(pred, points, joint_min, joint_range, threshold=0.6):
    """Keep graspable points and apply the learned point-to-TCP offset."""
    pred = {key: value[0].detach().cpu() for key, value in pred.items()}
    scores = F.softmax(pred['pred_graspable'], dim=1)[:, 1]
    mask = (pred['pred_graspable'].argmax(dim=1) == 1) & (scores > threshold)
    if not mask.any():
        return None
    scores, order = scores[mask].sort(descending=True)
    offsets = pred['pred_point_offset'][mask][order].numpy()
    rotations = compute_rotation_matrix_from_ortho6d(pred['pred_pose'][mask][order]).numpy()
    pre = pred['pred_joint_init'][mask][order].numpy().clip(0, 1) * joint_range + joint_min
    final = pred['pred_joint_final'][mask][order].numpy().clip(0, 1) * joint_range + joint_min
    return dict(score=scores.numpy(), position=points[mask.numpy()][order.numpy()] - offsets,
                rotation=rotations, pre_joints=pre, final_joints=final, offset=offsets)


def grasp_nms(grasps, distance=0.002, angle=np.deg2rad(5)):
    """Match the source example's distance AND rotation suppression criterion."""
    if grasps is None:
        return None
    keep = []
    order = np.argsort(grasps['score'])
    while order.size:
        current = order[-1]
        keep.append(current)
        other = order[:-1]
        distances = np.linalg.norm(grasps['position'][other] - grasps['position'][current], axis=1)
        relative = np.einsum('ij,nij->n', grasps['rotation'][current], grasps['rotation'][other])
        angles = np.arccos(np.clip((relative - 1) / 2, -1, 1))
        order = other[(distances > distance) & (angles > angle)]
    return {key: value[keep] for key, value in grasps.items()}


def load_cloud(rgb_path, depth_path, intrinsics, voxel_size=0.003):
    rgb_bgr = cv2.imread(str(rgb_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth is None or depth.ndim != 2:
        raise ValueError('Expected a readable RGB image and a single-channel depth PNG.')
    if rgb_bgr.shape[:2] != depth.shape:
        raise ValueError('RGB and depth images must have the same aligned resolution.')
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    points, colors = rgb_depth_to_pointcloud(depth, rgb, **intrinsics)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud.voxel_down_sample(voxel_size)


def predict(rgb_path, depth_path, checkpoint, intrinsics, num_points=20000, voxel_size=0.003):
    cloud = load_cloud(rgb_path, depth_path, intrinsics, voxel_size)
    points = np.asarray(cloud.points)
    if not len(points):
        raise ValueError('The depth image produced no points.')
    if len(points) > num_points:
        points = points[np.random.default_rng(0).choice(len(points), num_points, replace=False)]
    # The original example does not pad short clouds. Keep the same behavior.
    centered = points - points.mean(axis=0)
    model = PointTransformerV3(is_training=False).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    with torch.inference_mode():
        pred = model(dataloader_preprosess(torch.from_numpy(centered).unsqueeze(0).float()))
    cfg = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    joint_min = np.deg2rad(cfg['hand']['joints_min'])
    joint_range = np.deg2rad(cfg['hand']['joints_max']) - joint_min
    return grasp_nms(decode_predictions(pred, points, joint_min, joint_range))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb', type=Path, default=Path('examples/rgb.png'))
    parser.add_argument('--depth', type=Path, default=Path('examples/depth.png'))
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/dgs_net_best.pth'))
    parser.add_argument('--output', type=Path, default=Path('outputs/grasps.npz'))
    parser.add_argument('--fx', type=float, default=605.0880126953125)
    parser.add_argument('--fy', type=float, default=604.061279296875)
    parser.add_argument('--cx', type=float, default=323.97198486328125)
    parser.add_argument('--cy', type=float, default=240.3543701171875)
    parser.add_argument('--visualize-count', type=int, default=4,
                        help='Random post-NMS grasps to show in Open3D; 0 disables visualization')
    parser.add_argument('--visualize-seed', type=int,
                        help='Seed for reproducible visualization selection')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Inference requires a CUDA GPU and the CUDA extensions listed in README.md.')
    grasps = predict(args.rgb, args.depth, args.checkpoint,
                     dict(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **(grasps or {}))
    print(f'Saved {0 if grasps is None else len(grasps["score"])} grasps to {args.output}')
    if args.visualize_count > 0 and grasps is not None:
        from dgs_net.visualize import show_grasps

        cloud = load_cloud(args.rgb, args.depth,
                           dict(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy))
        show_grasps(cloud, grasps, count=args.visualize_count, seed=args.visualize_seed)


if __name__ == '__main__':
    main()
