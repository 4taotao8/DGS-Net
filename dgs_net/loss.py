
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_loss_and_acc(pred, gt):
    
    dist_thresh = 0.02
    para_type = pred['pred_graspable'].dtype
    device = pred['pred_graspable'].device

    # Candidate and reference points are both positive for graspability.
    pred_graspable = pred['pred_graspable'].permute(0, 2, 1)
    raw_gp_label = gt['grasp_points_mask'].to(device).contiguous()
    # raw_gp_label: [B, N], values are -1, 0, 1
    # 二分类标签：
    # -1 -> 0，不可抓取点，负样本
    #  0 -> 1，抓取候选点，正样本
    #  1 -> 1，抓取点，正样本
    gp_label = torch.zeros_like(raw_gp_label, dtype=torch.long)
    gp_label[raw_gp_label == 0] = 1
    gp_label[raw_gp_label == 1] = 1
    # 每个点自己的 loss 权重：
    # -1 -> 0.1
    #  0 -> 1.0
    #  1 -> 2.0
    point_weight = torch.ones_like(raw_gp_label, dtype=torch.float)
    point_weight[raw_gp_label == -1] = 0.1
    point_weight[raw_gp_label == 0] = 1.0
    point_weight[raw_gp_label == 1] = 2.0
    gp_loss_each = F.cross_entropy(
        pred_graspable,
        gp_label,
        reduction='none',
    )
    # Divide by the point count, as in the paper's point-loss equation.
    gp_loss = (gp_loss_each * point_weight).mean()
    out_gp = torch.argmax(pred_graspable, dim=1)
    gp_acc = two_class_acc(out_gp, gp_label)
    # 抓取位姿损失
    loss_pose = PoseLoss(weight = 0.5, angle_thresh=15)     
    gp_mask = torch.where(raw_gp_label > 0)
    pred_pose = pred['pred_pose']
    gt_pose = gt['grasp_pose'].to(device).contiguous()
    has_positive = gp_mask[0].numel() > 0
    if has_positive:
        _, _, pose_loss, pose_acc = loss_pose(pred_pose[gp_mask], gt_pose[gp_mask])
    else:
        pose_loss = pred_pose.sum() * 0.0
        pose_acc = pred_pose.new_zeros(())
    
    # 抓取点偏移损失
    loss_offset = RelativeL1Loss()
    gt_point_offset = gt['point_offset'].type(para_type).to(device).contiguous()
    pred_point_offset = pred['pred_point_offset']
    if has_positive:
        offset_loss = loss_offset(pred_point_offset[gp_mask], gt_point_offset[gp_mask])
        offset_acc_mask = torch.norm(pred_point_offset[gp_mask]-gt_point_offset[gp_mask], dim=-1) < dist_thresh
        offset_acc = offset_acc_mask.float().mean()
    else:
        offset_loss = pred_point_offset.sum() * 0.0
        offset_acc = pred_point_offset.new_zeros(())
    
    # 关节角度损失
    loss_joint_init = nn.MSELoss()
    loss_joint_final = nn.MSELoss()
    gt_joint_init, gt_joint_final=gt['grasp_joint_init'].type(para_type).to(device).contiguous(), gt['grasp_joint_final'].type(para_type).to(device).contiguous()
    pred_joint_init, pred_joint_final = pred['pred_joint_init'], pred['pred_joint_final']
    if has_positive:
        joint_init_loss = loss_joint_init(pred_joint_init[gp_mask], gt_joint_init[gp_mask])
        joint_final_loss = loss_joint_final(pred_joint_final[gp_mask], gt_joint_final[gp_mask])
    else:
        joint_init_loss = pred_joint_init.sum() * 0.0
        joint_final_loss = pred_joint_final.sum() * 0.0
    
    
    loss = {
        'gp_loss':              gp_loss,
        'pose_loss':            pose_loss,
        'offset_loss':          offset_loss,
        'joint_init_loss':      joint_init_loss,
        'joint_final_loss':     joint_final_loss,
        'total_loss':           gp_loss + 2*pose_loss + offset_loss + joint_init_loss + joint_final_loss
        }
    
    acc = {
        'TP':             gp_acc[0],
        'FP':             gp_acc[1],
        'TN':             gp_acc[2],
        'FN':             gp_acc[3],
        'gp_acc':         gp_acc[4],
        'gp_precision':   gp_acc[5],
        'gp_recall':      gp_acc[6],
        'gp_F1':          gp_acc[7],
        'pose_acc':       pose_acc,
        'offset_acc':     offset_acc,
        }
    
    return loss, acc



def two_class_acc(pred,gt):
    TP = torch.sum((pred == 1) & (gt == 1)).float()
    FP = torch.sum((pred == 1) & (gt == 0)).float()
    TN = torch.sum((pred == 0) & (gt == 0)).float()
    FN = torch.sum((pred == 0) & (gt == 1)).float()
    p = TP / (TP + FP).clamp_min(1)
    r = TP / (TP + FN).clamp_min(1)
    F1 = 2 * r * p / (r + p).clamp_min(1e-8)
    acc = (TP + TN) / (TP + TN + FP + FN)
    acc_list = [TP,FP,TN,FN,acc,p,r,F1]
    return acc_list  

class PoseLoss(nn.Module):
    def __init__(self, weight = 0.5, angle_thresh = 10):
        super(PoseLoss, self).__init__()
        if not (0 <= weight <= 1):
            raise ValueError(f"Weight must be between 0 and 1, but got {weight}")
        self.weight = weight
        self.angle_thresh = angle_thresh*np.pi/180

    def forward(self, pred, gt):
        if pred.shape[-1] != 6:
            raise ValueError(f"Expected pred_pose dimension to be 6, but got {pred.shape[-1]}")
        if gt.shape[-1] != 4:
            raise ValueError(f"Expected gt_pose dimension to be 4(quaternion), but got {pred.shape[-1]}")
        self.n = pred.shape[0]
        gt = gt.type(pred.dtype)

        # Rotation supervision uses the matrix MSE and geodesic terms below.

        pred_rmat = compute_rotation_matrix_from_ortho6d(pred)
        gt_rmat = compute_rotation_matrix_from_quaternion(gt)

        mse_loss = self.loss_MSE(pred_rmat,gt_rmat)
        geodesic_loss = self.loss_geodesic(pred_rmat,gt_rmat)
        acc = self.acc_geodesic()

        totle_loss = self.weight * mse_loss + (1 - self.weight) * geodesic_loss
        
        return mse_loss,geodesic_loss,totle_loss,acc
    
    def acc_geodesic(self):
        geodesic_mask = self.geodesic_loss < self.angle_thresh
        acc = torch.sum(geodesic_mask)/self.n
        return acc
        
    
    def loss_MSE(self, predict_rmat, gt_rmat):
        loss = torch.pow(gt_rmat - predict_rmat, 2).mean()
        return loss

    def loss_geodesic(self, predict_rmat, gt_rmat):
        rmat_diff = torch.bmm(predict_rmat.transpose(1, 2), gt_rmat)
        trace = torch.clamp(rmat_diff.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1), -1.0, 3.0)
        self.geodesic_loss = torch.acos(torch.clamp(((trace - 1) / 2), -0.999, 0.999))
        return self.geodesic_loss.mean()
    
def compute_rotation_matrix_from_quaternion(quaternion):
    n=quaternion.shape[0]
    quat = normalize_vector(quaternion)
    qx = quat[...,0].view(n, 1)
    qy = quat[...,1].view(n, 1)
    qz = quat[...,2].view(n, 1)
    qw = quat[...,3].view(n, 1)

    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #n*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #n*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #n*3
    matrix = torch.cat((row0.view(n, 1, 3), row1.view(n,1,3), row2.view(n,1,3)),1) #n*3*3
    return matrix

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#n*3
    y_raw = poses[:,3:6]#n*3
        
    x = normalize_vector(x_raw) #n*3
    z = cross_product(x,y_raw) #n*3
    z = normalize_vector(z)#n*3
    y = cross_product(z,x)#n*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #n*3*3
    return matrix

def normalize_vector(v):
    batch=v.shape[0]
    v_mag = torch.linalg.vector_norm(v, dim=1).clamp_min(1e-8)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

def cross_product(u, v):
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
    return out

class RelativeL1Loss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(RelativeL1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # 计算相对误差损失
        return torch.mean(torch.abs(y_pred - y_true) / (torch.abs(y_true) + self.epsilon)).float()


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)


def quaternion_to_ortho6d(quaternion):
    """
    Convert quaternion to 6D rotation representation.

    Args:
        quaternion: Tensor with shape [..., 4], quaternion order is [x, y, z, w]

    Returns:
        ortho6d: Tensor with shape [..., 6]
                 first 3 dims are the first column of rotation matrix,
                 last 3 dims are the second column of rotation matrix.
    """
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension to be 4, but got {quaternion.shape[-1]}")

    quat = F.normalize(quaternion, p=2, dim=-1, eps=1e-8)

    qx = quat[..., 0]
    qy = quat[..., 1]
    qz = quat[..., 2]
    qw = quat[..., 3]

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.stack((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), dim=-1)
    row1 = torch.stack((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), dim=-1)
    row2 = torch.stack((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), dim=-1)

    rotmat = torch.stack((row0, row1, row2), dim=-2)

    ortho6d = torch.cat(
        (
            rotmat[..., :, 0],
            rotmat[..., :, 1],
        ),
        dim=-1,
    )

    return ortho6d
