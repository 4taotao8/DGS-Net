"""Load point-level labels emitted by generate_training_data.py."""

from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import Dataset


CFG = yaml.safe_load((Path(__file__).resolve().parent.parent / 'config.yaml').read_text())


class GraspDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = Path(data_path) / 'scenes'
        self.frames_per_scene = CFG['num_image_per_scenes']
        bounds = CFG['dataset'][f'{split}_range']
        self.scene_ids = range(*bounds)
        self.joint_min = np.deg2rad(CFG['hand']['joints_min'])
        self.joint_range = np.deg2rad(CFG['hand']['joints_max']) - self.joint_min

    def __len__(self):
        return len(self.scene_ids) * self.frames_per_scene

    def __getitem__(self, index):
        scene_id = self.scene_ids[index // self.frames_per_scene]
        frame = index % self.frames_per_scene
        root = self.data_path / f'scene_{scene_id:04d}'
        pcd = np.load(root / 'pointcloud' / f'{frame:04d}.npy', allow_pickle=True).item()
        grasps = np.load(root / 'grasp' / f'{frame:04d}.npy', allow_pickle=True)
        points = pcd['points']
        point_data = {'points': points, 'norm_points': points - points.mean(axis=0)}
        offset = np.zeros((len(points), 3))
        pose = np.zeros((len(points), 4))
        pre = np.zeros((len(points), 20))
        final = np.zeros((len(points), 20))
        for grasp in grasps:
            point_id = grasp['grasp_points_id']
            offset[point_id] = points[point_id] - grasp['t7d'][-3:]
            pose[point_id] = grasp['t7d'][:4]
            pre[point_id] = (np.clip(grasp['pre_joints'], self.joint_min,
                                     self.joint_min + self.joint_range) - self.joint_min) / self.joint_range
            final[point_id] = (np.clip(grasp['exprct_joints'], self.joint_min,
                                       self.joint_min + self.joint_range) - self.joint_min) / self.joint_range
        targets = {
            'cls_mask': pcd['cls_mask'].astype(np.int64),
            'seg': pcd['seg'].astype(np.int64),
            'grasp_points_mask': pcd['grasp_points_mask'].astype(np.int64),
            'point_offset': offset,
            'grasp_pose': pose,
            'grasp_joint_init': pre,
            'grasp_joint_final': final,
            'scene_id': scene_id,
            'img_id': frame,
        }
        return point_data, targets
