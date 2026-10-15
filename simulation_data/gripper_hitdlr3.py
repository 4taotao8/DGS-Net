from pathlib import Path
import os
import time
import numpy as np
from numpy import sin,cos
import pybullet as p
import open3d as o3d
import enum

from . import btsim
from .transform import Rotation, Transform

# 食指      0   [[侧摆，指根，指尖，指尖],
# 中指      1   [侧摆，指根，指尖，指尖],
# 无名指    2   [侧摆，指根，指尖，指尖],
# 小指      3   [侧摆，指根，指尖，指尖],
# 拇指      4   [侧摆，指根，指尖，指尖]]

class Label(enum.IntEnum):
    FAILURE = 0  # grasp execution failed due to collision or slippage
    SUCCESS = 1  # object was successfully removed


class Grasp(object):
    """Grasp parameterized as pose and 15Dof joint angles of a HITDLR2 robot hand.
    joint_angles: [thumb_(1,2,3), index_, middle_, ring_, little_]
    
    TODO(mbreyer): clarify definition of grasp frame
    """

    def __init__(self, pose, pre_joints=None, exprct_joints=None, label=None, type = None):
        self.pre_joints = pre_joints
        self.pose = pose
        self.exprct_joints = exprct_joints
        self.label = label
        self.type = type
    
    def to_grasps_dict(self):
        rot_trans7d = self.pose.to_list() # [qx, qy, qz, qw, x, y, z]
        pre_joints = self.pre_joints.flatten()
        exprct_joints = self.exprct_joints.flatten() # np,(5*5,) little->thumb
        label = self.label

        data = {'t7d': rot_trans7d, #物体坐标系下
                'pre_joints':pre_joints,
                'exprct_joints':exprct_joints,
                'label': label,
                'type': self.type
        }

        return data

class Gripper(object):
    """Simulated HITDLR3 hand.
    joint1 pitch -20~20 degree;
    joint2 yaw 0~90 degree;
    joint3 yaw 0~90 degree;
    joint4 yaw 0~90 degree
    """

    def __init__(self, world):
        self.world = world
        data_root = Path(os.environ.get('DGS_DATA_ROOT', Path(__file__).resolve().parent.parent / 'data'))
        self.urdf_path = data_root / 'urdfs/hitdlr3/hand_right_simplify.urdf'
        # self.urdf_path = Path("../data/urdfs/hitdlr3/hand_right.urdf")
        # self.urdf_path = Path("../data/urdfs/hitdlr3/hand_right_convex_hull.urdf")

        self.max_hand_width = 0.105
        self.finger2or3_hand_width = 0.07
        self.finger_depth = 0.055
       
        self.T_body_tcp = Transform(Rotation.from_quat([0, 0.21676, 0, 0.97622]), [0.0421128868793952, 0, 0.129018711905046])
        self.T_tcp_body = self.T_body_tcp.inverse()
        self.T_body_tip_plane = Transform(Rotation.from_quat([0.040905, 0.21657, -0.0090827, 0.97537]), [0.0549919133451352, 0.00513118621536029, 0.156590170585173])
        self.T_tip_plane_body = self.T_body_tip_plane.inverse()
        
        self.T_body_index = Transform(Rotation.from_quat([0.52133, 0.52133, 0.47771, 0.47771]), [-0.010382, 0.037505, 0.061309])
        self.T_index_body = self.T_body_index.inverse()
        self.T_body_mid = Transform(Rotation.from_quat([0.52133, 0.52133, 0.47771, 0.47771]), [-0.010382, 0.012505, 0.071309])
        self.T_mid_body = self.T_body_mid.inverse()
        self.T_body_ring = Transform(Rotation.from_quat([0.49807, 0.54362, 0.49812, 0.45638]), [-0.0091687, -0.014221, 0.066309])
        self.T_ring_body = self.T_body_ring.inverse()
        self.T_body_little = Transform(Rotation.from_quat([0.47393, 0.56479, 0.51752, 0.43426]), [-0.0056537, -0.040608, 0.052309])
        self.T_little_body = self.T_body_little.inverse()
        self.T_body_thumb = Transform(Rotation.from_quat([0.46211, -0.19562, -0.45353, 0.73654]), [0.037455, 0.038919, 0.036487])
        self.T_thumb_body = self.T_body_thumb.inverse()
        
        self.T_tcp_3f_tcp=Transform(Rotation.identity(),[0,0.025,0.010])
        self.T_tcp_2f_tcp=Transform(Rotation.identity(),[0,0.0375,0.010])
        # self.palm_center_xyz = self.T_body_tcp.translation
        # self.T_palm_center_body = Transform(Rotation.from_euler('xyz',[0.0, 0.329867, 0.0]), [0.037026,0,0.13205])
        # self.T_body_palm_center = self.T_palm_center_body.inverse()

        self.joints_max_limits = np.array([0.0, 0.35, 1.48, 1.4, 1.4, 0,
                                           0.0, 0.35, 1.48, 1.4, 1.4, 0,
                                           0.0, 0.35, 1.48, 1.4, 1.4, 0,
                                           0.0, 0.35, 1.48, 1.4, 1.4, 0,
                                           0.0, 0.35, 1.48, 1.4, 1.4, 0,])
        self.joints_min_limits = np.array([-0.0, -0.35, 0, 0, 0, -0.0,
                                           -0.0, -0.35, 0, 0, 0, -0.0,
                                           -0.0, -0.35, 0, 0, 0, -0.0,
                                           -0.0, -0.35, 0, 0, 0, -0.0,
                                           -0.0, -0.35, 0, 0, 0, -0.0,]) 

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        # T_world_body = T_world_tcp
        self.body = self.world.load_urdf(self.urdf_path, T_world_body, selfcollision=True)
        # 删除掉没用的自碰撞
        self.remove_extra_selfcollision()
        self.body.set_pose(T_world_body)  # 设置的是urdf，visual的位置
        
        self.T_body_inertial = self.body.T_base_inertial
        T_world_inertial = T_world_body * self.T_body_inertial
        # constraint这东西设置的是惯性系的位置！！！！坑点！！！！
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            p.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_inertial,
        )
        p.changeVisualShape(self.body.uid, -1, rgbaColor=[0.660588235, 0.660588235, 0.660588235, 1])
        for i in range (30):
            p.changeVisualShape(self.body.uid, i, rgbaColor=[0.860588235, 0.860588235, 0.860588235, 1])
        
        # # 绘制坐标系
        # x = np.array([0.04, 0, 0, 1])
        # y = np.array([0, 0.04, 0, 1])
        # z = np.array([0, 0, 0.04, 1])
        # x_end = np.dot(T_world_tcp.as_matrix(),x).tolist()
        # y_end = np.dot(T_world_tcp.as_matrix(),y).tolist()
        # z_end = np.dot(T_world_tcp.as_matrix(),z).tolist()
        # origin = T_world_tcp.translation.tolist()
        # self.world.addUserDebugLine(origin, x_end[:3], [1,0,0], 4)
        # self.world.addUserDebugLine(origin, y_end[:3], [0,1,0], 4)
        # self.world.addUserDebugLine(origin, z_end[:3], [0,0,1], 4)
 
        return self.body.uid
    
    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        T_world_inertial = T_world_body * self.T_body_inertial
        self.constraint.change(
            jointChildPivot=T_world_inertial.translation,
            jointChildFrameOrientation=T_world_inertial.rotation.as_quat(),
            maxForce=300,
            erp=1
        )
    
    def remove_extra_selfcollision(self):
        robotId = self.body.uid
        # for ii in [-1,0,1,2,6,7,8,12,13,14,18,19,20,24,25,26]:
        for ii in [-1,0,1,2,3,6,7,8,9,12,13,14,15,18,19,20,21,24,25,26,27]:
            for jj in range(30):
                if ii == jj:
                    continue
                p.setCollisionFilterPair(robotId, robotId, ii, jj, enableCollision=0)
        for i in range(6):
            for j in range(i + 1, 6):
                p.setCollisionFilterPair(robotId, robotId, i, j, enableCollision=0)
        for i in range(6, 12):
            for j in range(i + 1, 12):
                p.setCollisionFilterPair(robotId, robotId, i, j, enableCollision=0)
        for i in range(12, 18):
            for j in range(i + 1, 18):
                p.setCollisionFilterPair(robotId, robotId, i, j, enableCollision=0)
        for i in range(18, 24):
            for j in range(i + 1, 24):
                p.setCollisionFilterPair(robotId, robotId, i, j, enableCollision=0)
        for i in range(24, 30):
            for j in range(i + 1, 30):
                p.setCollisionFilterPair(robotId, robotId, i, j, enableCollision=0)

    def set_tcp_pose(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)
        
    def get_tcp_pose(self):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp
        return T_world_tcp

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
                # print(self.get_tcp_pose().to_list())
            contact,a = self.detect_contact()
            if abort_on_contact and contact:
                return
        self.world.multi_step(10)

    def detect_contact(self, selfcontact=False):
        contacts=list(self.world.get_contacts(self.body))
        if selfcontact == False:
            contacts = [x for x in contacts if x[1] != x[2]]
            
        if contacts != []:
            return True, contacts
        else:
            return False, None
    
    def random_pre_grasp_pose(self,grasp_mathod):
        # 随机生成一个预抓取姿态
        # 食指      [[侧摆，指根，指尖，指尖],
        # 中指      [侧摆，指根，指尖，指尖],
        # 无名指    [侧摆，指根，指尖，指尖],
        # 小指      [侧摆，指根，指尖，指尖],
        if grasp_mathod == '5finger_circle' or  grasp_mathod == '5finger_rectangle':
            ch = np.random.choice([1,2,3])
            if ch == 1:
                joints = np.array([[0.24, 0.0, 0.0, 0.0],
                                   [0.08, 0.0, 0.0, 0.0],
                                   [-0.08, 0.0, 0.0, 0.0],
                                   [-0.24, 0.0, 0.0, 0.0],
                                   [0.12, 0.0, 0.0, 0.0]])
            elif ch == 2:
                joints = np.array([[0.12, 0.0, 0.0, 0.0],
                                   [0.04, 0.0, 0.0, 0.0],
                                   [-0.04, 0.0, 0.0, 0.0],
                                   [-0.12, 0.0, 0.0, 0.0],
                                   [0.12, 0.0, 0.0, 0.0]])
            elif ch == 3:
                joints = np.array([[0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0]])
        elif grasp_mathod == '3finger':
            ch = np.random.choice([1,2,3])
            if ch == 1:
                joints = np.array([[0.010, 0.0, 0.0, 0.0],
                                [0.05, 0.0, 0.0, 0.0],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.05, 0, 0, 0]])
            elif ch == 2:
                joints = np.array([[0.06, 0.0, 0.0, 0.0],
                                [0.03, 0.0, 0.0, 0.0],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.03, 0.0, 0.0, 0.0]])
            elif ch == 3:
                joints = np.array([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 0.0, 0.0, 0.0]])
        elif grasp_mathod == '2finger':
            ch = np.random.choice([1,2,3])
            if ch == 1:
                joints = np.array([[0.05, 0.0, 0.0, 0.0],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 1.3, 1.3, 1.3],
                                [-0.05, 0, 0, 0]])
            elif ch == 2:
                joints = np.array([[0.03, 0.0, 0.0, 0.0],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 1.3, 1.3, 1.3],
                                [-0.03, 0.0, 0.0, 0.0]])
            elif ch == 3:
                joints = np.array([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 1.3, 1.3, 1.3],
                                [0.0, 0.0, 0.0, 0.0]])
        self.set_joint_pose(joints) 
    
    def set_joint_pose(self,joints):
        # 去掉固定关节
        joints_min_limits = self.joints_min_limits.reshape(5, 6)[:, 1:-1]  
        joints_max_limits = self.joints_max_limits.reshape(5, 6)[:, 1:-1]
        joints = np.clip(joints, joints_min_limits, joints_max_limits)
        # 食指      [[侧摆，指根，指尖，指尖],
        # 中指      [侧摆，指根，指尖，指尖],
        # 无名指    [侧摆，指根，指尖，指尖],
        # 小指      [侧摆，指根，指尖，指尖],
        # 拇指      [侧摆，指根，指尖，指尖]]
        for finger in range(5):
            for i in range(4):
                p.setJointMotorControl2(self.body.uid, finger * 6 + i+1, p.POSITION_CONTROL, targetPosition=joints[finger][i])
        self.world.multi_step(10)   
        
    def read_joint_pose(self):
        # read joint positions
        # 食指      [[侧摆，指根，指尖，指尖],
        # 中指      [侧摆，指根，指尖，指尖],
        # 无名指    [侧摆，指根，指尖，指尖],
        # 小指      [侧摆，指根，指尖，指尖],
        # 拇指      [侧摆，指根，指尖，指尖]]
        jointPositions = []
        for i in range(p.getNumJoints(self.body.uid)):
            jointPositions.append(p.getJointState(self.body.uid, i)[0])
        jointPositions=np.array(jointPositions).reshape(5,6)[:,1:-1]
        return jointPositions

    def admittance_control(self, pos):
        # pos是5*4的，这里加上固定关节变成5*6
        jpos = np.insert(pos, 0, 0, axis=1)
        jpos = np.insert(jpos, 5, 0, axis=1)
        jpos = jpos.flatten()
        # 初始化导纳控制参数和状态变量
        # Kdi = 0.0
        # Kdp = 0.8
        # Kst = 0.2
        t_ms = 0.001
        kd_list = [[[0, 0., 0.], # fix
                   [0, 0.9,1.9], # yaw
                   [0, 0.9,0.4], # pitch
                   [0, 0.9,0.4], # pitch
                   [0, 0.9,0.4], # pitch
                   [0, 0., 0.]], # fix   index

                   [[0, 0.,0.],
                   [0, 0.9,0.9],
                   [0, 0.9,0.2],
                   [0, 0.9,0.2],
                   [0, 0.9,0.2],
                   [0, 0.,0.]],  # mid

                   [[0, 0.,0.],
                   [0, 0.9,0.9],
                   [0, 0.9,0.2],
                   [0, 0.9,0.2],
                   [0, 0.9,0.2],
                   [0, 0.,0.]],  # ring

                   [[0, 0.,0.],
                   [0, 0.9,0.9],
                   [0, 0.9,0.2],
                   [0, 0.9,0.2],
                   [0, 0.9,0.2],
                   [0, 0.,0.]],  # little

                   [[0, 0.,0.],
                   [0, 1.9,15],
                   [0, 1.9,1.9],
                   [0, 1.9,6.9],
                   [1, 1.9,6.9],
                   [0, 0.,0.]]]  # thumb

        # 控制循环
        num_steps = 1000
        xas = [[0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0]]  # 存储关节位置的列表
        vxas = [[0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0]]  # 存储关节位置的列表
        
        pre_joints = []
        for ii in range(p.getNumJoints(self.body.uid)):
            pre_joints.append(p.getJointState(self.body.uid, ii)[0])
        pre_joints = np.array(pre_joints)

        joint_torq_before = 0
        for i in range(num_steps):  # 执行1000个控制周期
            # 遍历每个手指和关节         
            for finger in range(5):
                for joint in range(6):

                    if joint == 0 or joint == 5:
                        continue  # 跳过关节0，因为它是固定的（fix）

                    Kdi = kd_list[finger][joint][0]
                    Kdp = kd_list[finger][joint][1]
                    Kst = kd_list[finger][joint][2]
                    
                    # 获取当前手指和关节的位置和速度
                    desire_xa = pre_joints[finger * 6 + joint] + (i+1) * (jpos[finger * 6 + joint] - pre_joints[finger * 6 + joint]) / num_steps
                    vxa = p.getJointState(self.body.uid, finger * 6 + joint)[1]
                    joint_torq = p.getJointState(self.body.uid, finger * 6 + joint)[3]
                    
                    # 打印出指尖关节的力矩
                    # if joint == 4:
                    #     if joint_torq  > joint_torq_before:
                    #         joint_torq_before = joint_torq
                            # print (joint_torq_before)

                    # 计算导纳控制算法
                    if Kdi != 0.0:
                        axa = (1 / Kdi) * (joint_torq - Kdp * vxa - Kst * xas[finger][joint])
                        vxa = axa * t_ms + vxa
                        xas[finger][joint] = vxa * t_ms + xas[finger][joint]
                    elif Kdp != 0.0:
                        axa = 0
                        vxas[finger][joint] = (1 / Kdp) * (joint_torq - Kst * xas[finger][joint])
                        xas[finger][joint] = vxas[finger][joint] * t_ms + xas[finger][joint]
                    elif Kst != 0.0:
                        axa = 0
                        vxa = 0
                        xa = joint_torq / Kst
                    else:
                        axa = 0
                        vxa = 0
                        xa = 0

                    xa_set = desire_xa-xas[finger][joint]
                    xa_set = min(xa_set, self.joints_max_limits[finger * 6 + joint])
                    xa_set = max(xa_set, self.joints_min_limits[finger * 6 + joint])

                    if joint == 3 or joint == 4:
                        # 将关节3和关节4的角度设置为相同
                        p.setJointMotorControl2(self.body.uid, finger * 6 + 3, p.POSITION_CONTROL, targetPosition=xa_set)
                        p.setJointMotorControl2(self.body.uid, finger * 6 + 4, p.POSITION_CONTROL, targetPosition=xa_set)
                    else:
                        # 更新机器人关节位置
                        p.setJointMotorControl2(self.body.uid, finger * 6 + joint, p.POSITION_CONTROL, targetPosition=xa_set)
            # 模拟仿真步长
            p.stepSimulation()

    def conscircle(self, grasp_method ='5finger_circle', height_mode=None):
        # 返回值：五个手指的指尖坐标（tip_plane下），顺序：index,mid,ring,little,thumb
        if grasp_method == '5finger_circle':
            n = 5
            if height_mode is None:
                mode = np.random.randint(1, 4)
            else:
                mode = height_mode
            if mode==1:
                r = 0.015
                z = 0.0003
                ts = np.array([125, 170, 205, 250])*np.pi/180.0
                ts = np.insert(ts, 4, 15*np.pi/180.0)
                xs = r * cos(ts)
                ys = r * sin(ts)*2
                zs = [z-0.0005, z+0.008, z+0.006, z-0.0015, z-0.002]
            elif mode==2:
                r = 0.008
                z = -0.01
                ts = np.array([125, 170, 205, 250])*np.pi/180.0
                ts = np.insert(ts, 4, 15*np.pi/180.0)
                xs = r * cos(ts)
                ys = r * sin(ts)*4
                zs = [z-0.005, z+0.008, z+0.006, z-0.008, z-0.005]
            elif mode==3:
                r = 0.0001
                z = -0.02
                ts = np.linspace(120.0, 240.0, n)*np.pi/180.0
                ts = np.insert(ts, 4, 20*np.pi/180.0)
                xs = [0.01,0,0,0.015,0]
                ys = r * sin(ts)*4
                zs = [z-0.01, z+0.005, z+0.006, z-0.015, z-0.001]
            else:
                print('[error]: height_mode error!')
                raise NotImplementedError
            tip_p = [[x,y,z] for x, y, z in zip(xs, ys, zs)]
            return np.array(tip_p)
        
        elif grasp_method == '5finger_rectangle':
            n = 5
            if height_mode is None:
                mode = np.random.randint(1, 4)
            else:
                mode = height_mode
            if mode==1:
                r = 0.006
                z = 0.0
                xs = [-r, -r, -r, -r, r*0.8]
                ys = [0.037, 0.010, -0.016, -0.037, 0.005]
                zs = [z, z+0.01, z+0.01, z, z-0.002]
            elif mode==2:
                r = 0.004
                z = -0.005
                xs = [-r, -r, -r, -r, r*0.8]
                ys = [0.037, 0.010, -0.016, -0.037, 0.005]
                zs = [z, z+0.01, z+0.01, z, z-0.002]
            elif mode==3:
                r = 0.000001
                z = -0.01
                xs = [-r*0.95, -r, -r, -r*0.9, r*0.8]
                ys = [0.037, 0.010, -0.016, -0.037, 0.005]
                zs = [z, z+0.01, z+0.01, z, z-0.002]
            else:
                print('[error]: height_mode error!')
                raise NotImplementedError
            tip_p = [[x,y,z] for x, y, z in zip(xs, ys, zs)]
            return np.array(tip_p)
        
        elif grasp_method == '3finger':
            # index, mid, thumb
            if height_mode is None:
                mode = np.random.randint(1, 4)
            else:
                mode = height_mode
            if mode==1:
                r = 0.004
                z = 0.00
                xs = [-r, -r*0.7, r*0.8]
                ys = [0.035, 0.009, 0.02]
                zs = [z, z+0.008, z-0.005]
            elif mode==2:
                r = 0.0025
                z = -0.008
                xs = [-r, -r*0.7, r*0.8]
                ys = [0.035, 0.009, 0.02]
                zs = [z, z+0.008, z-0.005]
            elif mode==3:
                r = 0.000001
                z = -0.016
                xs = [-r, -r*0.7, r*0.8]
                ys = [0.035, 0.009, 0.02]
                zs = [z, z+0.008, z-0.005]
            else:
                print('[error]: height_mode error!')
                raise NotImplementedError
            tip_p = [[x,y,z] for x, y, z in zip(xs, ys, zs)]
            return np.array(tip_p)
        
        elif grasp_method == '2finger':
            # index, thumb
            if height_mode is None:
                mode = np.random.randint(1, 4)
            else:
                mode = height_mode
            if mode==1:
                r = 0.004
                z = 0.00
                xs = [-r, r*0.8]
                ys = [0.035, 0.033]
                zs = [z, z-0.005]
            elif mode==2:
                r = 0.0025
                z = -0.008
                xs = [-r, r*0.8]
                ys = [0.035, 0.030]
                zs = [z, z-0.005]
            elif mode==3:
                r = 0.000001
                z = -0.016
                xs = [-r, r*0.8]
                ys = [0.035, 0.030]
                zs = [z, z-0.005]
            else:
                print('[error]: height_mode error!')
                raise NotImplementedError
            tip_p = [[x,y,z] for x, y, z in zip(xs, ys, zs)]
            return np.array(tip_p)  
            
    def graspit(self, grasp_method=None, height_mode=None ,joints=None):
        if joints is not None:
            jpos = joints
        else:
            if height_mode is None:
                height_mode = np.random.randint(1, 4)
            jpos = []
            finger_close_jpos = [0.0, 1.3, 1.3, 1.3]
            if grasp_method=='5finger_circle' or grasp_method=='5finger_rectangle':
                tip_ps = self.conscircle(grasp_method,height_mode)
                for i in range(5):
                    tip_f = self.Tipplane2Fingerbase(i,tip_ps[i])
                    jpos.append(self.FingerInverseKin(tip_f))
                jpos = np.array(jpos)
                # print(jpos)
                
            elif grasp_method=='3finger':
                tip_ps = self.conscircle(grasp_method,height_mode)
                for i,finger_id in enumerate([0,1,4]): # 0:index, 1:mid, 4:thumb
                    tip_f = self.Tipplane2Fingerbase(finger_id,tip_ps[i])
                    jpos.append(self.FingerInverseKin(tip_f))
                jpos = np.array(jpos)
                for i in [2,3]:   # 插入其他手指关节角度，变成闭合
                    jpos = np.insert(jpos, i , finger_close_jpos, axis=0)
                # print(jpos)
                
            elif grasp_method=='2finger':
                tip_ps = self.conscircle(grasp_method,height_mode)
                for i,finger_id in enumerate([0,4]): # 0:index, 1:mid, 4:thumb
                    tip_f = self.Tipplane2Fingerbase(finger_id,tip_ps[i])
                    jpos.append(self.FingerInverseKin(tip_f))
                jpos = np.array(jpos)
                for i in [1,2,3]:   # 插入其他手指关节角度，变成闭合
                    jpos = np.insert(jpos, i , finger_close_jpos, axis=0)
                # print(jpos)
                
            else:
                print('[error]: grasp_method error!')
                raise NotImplementedError

        self.admittance_control(jpos)
        return jpos

    # def RightHand2FingerPosition(self, iFinger, tipP):
    #     # 0: thumb, 1: index, 2: middle, 3: ring, 4: little
    #     tipF = np.zeros(shape=(3,))
    #     if iFinger==0:
    #         tipF[0] = tipP[0]*(-0.69987) + tipP[1]*(-0.54959) + tipP[2]*(0.456218)    + (31.822) 
    #         tipF[1] = tipP[0]*(0.571046) + tipP[1]*(-0.8142) + tipP[2]*(-0.10481) + (8.91169) 
    #         tipF[2] = tipP[0]*(0.429052)    + tipP[1]*(0.187171)    + tipP[2]*(0.883675)    + (-105.284)
    #     elif iFinger==1:
    #         tipF[0] = tipP[0]*0.9962 + tipP[1]*(-0.0872) + 5.7292 
    #         tipF[1] = tipP[0]*0.0872 + tipP[1]*0.9962    - 36.4390 
    #         tipF[2] = tipP[2] - 108.0435 
    #     elif iFinger==2: # 2
    #         tipF[0] = tipP[0] + 3.7000 
    #         tipF[1] = tipP[1] - 10.0000 
    #         tipF[2] = tipP[2] - 118.0435 
    #     elif iFinger==3: # 3
    #         tipF[0] = tipP[0]*0.9962    + tipP[1]*0.0872 + 3.9852 
    #         tipF[1] = tipP[0]*(-0.0872) + tipP[1]*0.9962 + 16.5154 
    #         tipF[2] = tipP[2] - 113.0435 
    #     elif iFinger==4:
    #         tipF[0] = tipP[0]*0.9848    + tipP[1]*0.1736 + 6.5770 
    #         tipF[1] = tipP[0]*(-0.1736) + tipP[1]*0.9848 + 42.9067 
    #         tipF[2] = tipP[2] - 94.0435
    #     return tipF 

    def FingerJacobian(self,jpos):
        # jpos[侧摆，指根，指尖]
        a=jpos[0]
        b=jpos[1]
        c=jpos[2]
        dx_da = -0.00236246478472611*sin(a - b) + 0.00236246478472611*sin(a + b) + 0.00383737690337042*sin(-a + b + c) - 0.00469683404559205*sin(-a + b + 2*c) + 0.00383737690337042*sin(a + b + c) - 0.00469683404559205*sin(a + b + 2*c) + 0.0258908083588081*cos(a - b) + 0.0258908083588081*cos(a + b) + 0.0134639923322906*cos(-a + b + c) + 0.00798450938509482*cos(-a + b + 2*c) + 1.43722857830531e-18*cos(a - b + c) + 1.43722857830531e-18*cos(a + b - c) + 0.0134639923322906*cos(a + b + c) + 0.00798450938509482*cos(a + b + 2*c)
        dx_db = 0.00236246478472611*sin(a - b) + 0.00236246478472611*sin(a + b) - 0.00383737690337042*sin(-a + b + c) + 0.00469683404559205*sin(-a + b + 2*c) + 0.00383737690337042*sin(a + b + c) - 0.00469683404559205*sin(a + b + 2*c) - 0.0258908083588081*cos(a - b) + 0.0258908083588081*cos(a + b) - 0.0134639923322906*cos(-a + b + c) - 0.00798450938509482*cos(-a + b + 2*c) - 1.43722857830531e-18*cos(a - b + c) + 1.43722857830531e-18*cos(a + b - c) + 0.0134639923322906*cos(a + b + c) + 0.00798450938509482*cos(a + b + 2*c)
        dx_dc = -0.00383737690337042*sin(-a + b + c) + 0.0093936680911841*sin(-a + b + 2*c) + 0.00383737690337042*sin(a + b + c) - 0.0093936680911841*sin(a + b + 2*c) - 0.0134639923322906*cos(-a + b + c) - 0.0159690187701896*cos(-a + b + 2*c) + 1.43722857830531e-18*cos(a - b + c) - 1.43722857830531e-18*cos(a + b - c) + 0.0134639923322906*cos(a + b + c) + 0.0159690187701896*cos(a + b + 2*c)
        dy_da = -0.0258908083588081*sin(a - b) - 0.0258908083588081*sin(a + b) + 0.0134639923322906*sin(-a + b + c) + 0.00798450938509482*sin(-a + b + 2*c) - 1.73472347597681e-18*sin(a - b + c) - 1.73472347597681e-18*sin(a + b - c) - 0.0134639923322906*sin(a + b + c) - 0.00798450938509482*sin(a + b + 2*c) - 0.00236246478472611*cos(a - b) + 0.00236246478472611*cos(a + b) - 0.00383737690337042*cos(-a + b + c) + 0.00469683404559205*cos(-a + b + 2*c) + 0.00383737690337042*cos(a + b + c) - 0.00469683404559205*cos(a + b + 2*c)
        dy_db = 0.0258908083588081*sin(a - b) - 0.0258908083588081*sin(a + b) - 0.0134639923322906*sin(-a + b + c) - 0.00798450938509482*sin(-a + b + 2*c) + 1.73472347597681e-18*sin(a - b + c) - 1.73472347597681e-18*sin(a + b - c) - 0.0134639923322906*sin(a + b + c) - 0.00798450938509482*sin(a + b + 2*c) + 0.00236246478472611*cos(a - b) + 0.00236246478472611*cos(a + b) + 0.00383737690337042*cos(-a + b + c) - 0.00469683404559205*cos(-a + b + 2*c) + 0.00383737690337042*cos(a + b + c) - 0.00469683404559205*cos(a + b + 2*c)
        dy_dc = -0.0134639923322906*sin(-a + b + c) - 0.0159690187701896*sin(-a + b + 2*c) - 1.73472347597681e-18*sin(a - b + c) + 1.73472347597681e-18*sin(a + b - c) - 0.0134639923322906*sin(a + b + c) - 0.0159690187701896*sin(a + b + 2*c) + 0.00383737690337042*cos(-a + b + c) - 0.0093936680911841*cos(-a + b + 2*c) + 0.00383737690337042*cos(a + b + c) - 0.0093936680911841*cos(a + b + 2*c)
        dz_da = 0
        dz_db = 0.00472492956945222*sin(b) + 0.00767475380674084*sin(b + c) - 0.00939366809118411*sin(b + 2*c) + 0.0517816167176163*cos(b) + 0.0269279846645811*cos(b + c) + 0.0159690187701896*cos(b + 2*c)
        dz_dc = 0.00767475380674084*sin(b + c) - 0.0187873361823682*sin(b + 2*c) + 0.0269279846645811*cos(b + c) + 0.0319380375403792*cos(b + 2*c)
        jacobian_matrix = np.array([[dx_da, dx_db, dx_dc],
                                [dy_da, dy_db, dy_dc],
                                [dz_da, dz_db, dz_dc]])
        return jacobian_matrix
    
    # 手指逆运动学解析解
    def FingerInverseKin(self, tipF):
        # tipF期望指尖位置，单位m
        tipF[0] = np.clip(tipF[0], -0.033, 0.033)
        tipF[1] = np.clip(tipF[1], 0.0065, 0.13)
        tipF[2] = np.clip(tipF[2], 0.009, 0.11)
        tipF = [x for x in tipF]
        # 初始角度，单位弧度制
        Ang = np.zeros(shape=(3,))
        Ang[0] = 0        # 侧摆
        Ang[1] = 0.4      # 指根
        Ang[2] = 0.6      # 指尖
        # 雅各比
        Jac = self.FingerJacobian(Ang)
        Jac_inv = np.linalg.inv(Jac)
        # 求解
        for i in range(12):
            now_xyz = self.FingerDirectKin(Ang).translation
            delt_xyz = (tipF - now_xyz).reshape(3,1)
            if np.linalg.norm(delt_xyz) < 0.0001:
                break
            delt_Ang = Jac_inv @ delt_xyz
            Ang = Ang.reshape(3,1) + delt_Ang
            Jac = self.FingerJacobian(Ang.reshape(3))
            Jac_inv = np.linalg.inv(Jac)
        # 返回手指四个关节的角度，[侧摆，指根，指尖，指尖]
        # 限制角度范围
        Ang[0] = np.clip(Ang[0], self.joints_min_limits[1], self.joints_max_limits[1])
        Ang[1] = np.clip(Ang[1], self.joints_min_limits[2], self.joints_max_limits[2])
        Ang[2] = np.clip(Ang[2], self.joints_min_limits[3], self.joints_max_limits[3])

        return np.array([Ang[0],Ang[1],Ang[2],Ang[2]]).reshape(4,)
    
    def FingerDirectKin(self, jpos):
        # 手指坐标系，单位m
        # jpos[侧摆，指根，指尖]
        # 返回T_link_link5
        T_link_link1 = Transform(Rotation.identity(),[0, 0.02627, 0.012021])*Transform(Rotation.from_euler('z',-jpos[0]),[0, 0, 0])
        T_link1_link2 = Transform(Rotation.from_quat([0.47813, 0.52095, 0.47813, 0.52095]),[0,0,0])*Transform(Rotation.from_euler('z',jpos[1]),[0, 0, 0])
        T_link2_link3 = Transform(Rotation.from_quat([0, 0, 0.042819, 0.99908]),[0.051996, -0.00027708, 0])*Transform(Rotation.from_euler('z',jpos[2]),[0, 0, 0])
        T_link3_link4 = Transform(Rotation.identity(),[0.026928, -0.0076747, 0])*Transform(Rotation.from_euler('z',jpos[2]),[0, 0, 0])
        T_link4_link5 = Transform(Rotation.identity(),[0.015969, 0.0093937, 0])
        T_link_link5 =T_link_link1*T_link1_link2*T_link2_link3*T_link3_link4*T_link4_link5
        return T_link_link5
    
    def draw_finger_workspace(self):
        joint_ranges = {'a': np.linspace(self.joints_min_limits[1], self.joints_max_limits[1], 30),  # 侧摆关节角度范围
                        'b': np.linspace(self.joints_min_limits[2], self.joints_max_limits[2], 30),  # 指根关节角度范围
                        'c': np.linspace(self.joints_min_limits[3], self.joints_max_limits[3], 30),}  # 指尖关节角度范围
        workspace_points = []
        for a in joint_ranges['a']:
            for b in joint_ranges['b']:
                for c in joint_ranges['c']:
                    jpos = [a, b, c]
                    T = self.FingerDirectKin(jpos)
                    # 获取末端执行器位置 (T的最后一列即为末端位置)
                    end_effector_position = T.translation
                    workspace_points.append(end_effector_position)
        # 绘制工作空间
        workspace_points = np.array(workspace_points)
        import matplotlib.pyplot as plt

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2], c='b', marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        x_min, y_min, z_min = np.min(workspace_points, axis=0)
        x_max, y_max, z_max = np.max(workspace_points, axis=0)
        print(f"工作空间的范围：")
        print(f"X轴范围: [{x_min}, {x_max}]")
        print(f"Y轴范围: [{y_min}, {y_max}]")
        print(f"Z轴范围: [{z_min}, {z_max}]")
        plt.title('Finger Workspace')
        plt.show()
        
    def Tipplane2Fingerbase(self, i_finger, tip_p):
        tip_p = np.array(tip_p)
        tip_b = self.T_body_tip_plane.transform_point(tip_p)
        if i_finger == 0: # index
            tip_f = self.T_index_body.transform_point(tip_b)
        elif i_finger == 1: # mid
            tip_f = self.T_mid_body.transform_point(tip_b)
        elif i_finger == 2: # ring
            tip_f = self.T_ring_body.transform_point(tip_b)
        elif i_finger == 3: # little
            tip_f = self.T_little_body.transform_point(tip_b)
        elif i_finger == 4: # thumb
            tip_f = self.T_thumb_body.transform_point(tip_b)
        else:
            print("finger index error")
        return tip_f
    
    def extract_mesh_to_o3d(self,score = None,color=None):
        hand_id = self.body.uid
        meshes = []
        visual_data = p.getVisualShapeData(hand_id)
        if score is None:
            random_color = np.random.rand(3)
        else:
            # score=min((score-0.7)/0.15, 1.0)
            if color is not None:
                random_color = color
            else:
                random_color = np.array([176/255, 196/255, 210/255])
        for v in visual_data:
            link_index = v[1]  # 链接的索引
            mesh_filename = v[4].decode('utf-8')  # 获取STL文件路径
            if mesh_filename.endswith(".stl") or mesh_filename.endswith(".STL"):  # 处理STL文件
                # 加载STL网格并转换为Open3D格式
                if os.path.exists(mesh_filename):
                    mesh = o3d.io.read_triangle_mesh(mesh_filename)
                    mesh.compute_vertex_normals()

                    mesh.paint_uniform_color(random_color)

                    # 获取链接的变换信息
                    if link_index == -1:  # base_link
                        position, orientation = p.getBasePositionAndOrientation(hand_id)
                        base_info = p.getDynamicsInfo(hand_id, link_index)
                        T_world_mass = np.eye(4)
                        T_world_mass[:3, :3] = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
                        T_world_mass[:3, 3] = np.array(position)
                        T_base_mass = np.eye(4)
                        T_base_mass[:3, :3] = np.array(p.getMatrixFromQuaternion(base_info[4])).reshape(3, 3)
                        T_base_mass[:3, 3] = np.array(base_info[3])
                        T_mass_base = np.linalg.inv(T_base_mass)
                        transform = T_world_mass @ T_mass_base

                    else:
                        link_state = p.getLinkState(hand_id, link_index)
                        # link_state 的返回值包含多种信息，我们仅关心位置和姿态
                        if len(link_state) >= 6:
                            position, orientation = link_state[4], link_state[5]
                            transform = np.eye(4)
                            transform[:3, :3] = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
                            transform[:3, 3] = np.array(position)
                        else:
                            continue  # 忽略没有位姿信息的链接

                    # 对网格应用变换
                    mesh.transform(transform)
                    meshes.append(mesh)
        return meshes

    def extract_mesh_to_o3d_trans(self, score=None, color=None, alpha=1.0):
        hand_id = self.body.uid
        meshes = []
        materials = []
        visual_data = p.getVisualShapeData(hand_id)
        
        if score is None:
            rgb = np.random.rand(3)
        else:
            if color is not None:
                rgb = np.array(color[:3])  # 只取前3位
            else:
                rgb = np.array([176/255, 196/255, 210/255])
        
        for v in visual_data:
            link_index = v[1]
            mesh_filename = v[4].decode('utf-8')
            if mesh_filename.endswith(".stl") or mesh_filename.endswith(".STL"):
                if os.path.exists(mesh_filename):
                    mesh = o3d.io.read_triangle_mesh(mesh_filename)
                    mesh.compute_vertex_normals()
                    mesh.paint_uniform_color(rgb)

                    if link_index == -1:
                        position, orientation = p.getBasePositionAndOrientation(hand_id)
                        base_info = p.getDynamicsInfo(hand_id, link_index)
                        T_world_mass = np.eye(4)
                        T_world_mass[:3, :3] = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
                        T_world_mass[:3, 3] = np.array(position)
                        T_base_mass = np.eye(4)
                        T_base_mass[:3, :3] = np.array(p.getMatrixFromQuaternion(base_info[4])).reshape(3, 3)
                        T_base_mass[:3, 3] = np.array(base_info[3])
                        transform = T_world_mass @ np.linalg.inv(T_base_mass)
                    else:
                        link_state = p.getLinkState(hand_id, link_index)
                        if len(link_state) >= 6:
                            position, orientation = link_state[4], link_state[5]
                            transform = np.eye(4)
                            transform[:3, :3] = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
                            transform[:3, 3] = np.array(position)
                        else:
                            continue

                    mesh.transform(transform)
                    meshes.append(mesh)
                    
                    # 构建材质
                    mat = o3d.visualization.rendering.MaterialRecord()
                    mat.shader = "defaultLitTransparency"
                    mat.base_color = [rgb[0], rgb[1], rgb[2], alpha]  # RGBA
                    materials.append(mat)
        
        return meshes, materials

def draw_bounding_box(body_id, lifetime=1):
    """
    绘制指定物体的最小包围框 (AABB, Axis-Aligned Bounding Box)

    :param body_id: 物体的 ID
    :param lifetime: 线段的生存时间（秒），默认为 0.1 秒，可用于实时更新
    """
    # 获取物体的 AABB（min 和 max 角点）
    aabb_min, aabb_max = p.getAABB(body_id)
    
    # 计算 AABB 的 8 个角点
    corners = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
    ])

    # 定义 AABB 的 12 条边（起点、终点）
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底部四条边
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶部四条边
        (0, 4), (1, 5), (2, 6), (3, 7)   # 连接上下的四条边
    ]

    # 绘制 AABB
    for start, end in edges:
        p.addUserDebugLine(
            lineFromXYZ=corners[start],
            lineToXYZ=corners[end],
            lineColorRGB=[0.3, 0.3, 0.3], 
            lineWidth=2,
            lifeTime=lifetime  # 设置生存时间，使其可以动态更新
        )

def draw_bounding_cylinder(body_id, lifetime=1):
    """
    绘制指定物体的最小包围圆柱 (Axis-Aligned Bounding Cylinder)

    :param body_id: 物体的 ID
    :param lifetime: 线段的生存时间（秒），默认为 0.1 秒，可用于实时更新
    """
    # 获取物体的 AABB（min 和 max 角点）
    aabb_min, aabb_max = p.getAABB(body_id)
    
    # 计算圆柱的中心和半径
    # 中心是AABB的中心点
    center = np.array([
        (aabb_min[0] + aabb_max[0]) / 2,
        (aabb_min[1] + aabb_max[1]) / 2,
        (aabb_min[2] + aabb_max[2]) / 2
    ])

    # 圆柱的半径是AABB在x-y平面上的半径，假设AABB是轴对齐的
    # radius = np.mean(np.array([aabb_max[0] - aabb_min[0], aabb_max[1] - aabb_min[1]])) / 2
    radius = 0.02176

    # 圆柱的高度是AABB在z轴方向的大小
    height = aabb_max[2] - aabb_min[2]
    
    p.addUserDebugLine(
        lineFromXYZ=center - np.array([0, 0, height/2]), 
        lineToXYZ=center + np.array([0, 0, height/2]),
        lineColorRGB=[0.3, 0.3, 0.3],  
        lineWidth=2,
        lifeTime=lifetime
    )
    
    # 绘制圆柱底面和顶面
    num_points = 20
    for i in range(num_points):
        angle1 = 2 * np.pi * i / num_points
        angle2 = 2 * np.pi * (i + 1) / num_points
        # 底面圆的点
        bottom1 = center + np.array([radius * np.cos(angle1), radius * np.sin(angle1), -height/2])
        bottom2 = center + np.array([radius * np.cos(angle2), radius * np.sin(angle2), -height/2])
        # 顶面圆的点
        top1 = center + np.array([radius * np.cos(angle1), radius * np.sin(angle1), height/2])
        top2 = center + np.array([radius * np.cos(angle2), radius * np.sin(angle2), height/2])

        # 绘制底面边缘
        p.addUserDebugLine(
            lineFromXYZ=bottom1, 
            lineToXYZ=bottom2, 
            lineColorRGB=[0.3, 0.3, 0.3],  # 绿色
            lineWidth=2,
            lifeTime=lifetime
        )

        # 绘制顶面边缘
        p.addUserDebugLine(
            lineFromXYZ=top1, 
            lineToXYZ=top2, 
            lineColorRGB=[0.3, 0.3, 0.3],  # 绿色
            lineWidth=2,
            lifeTime=lifetime
        )

        # 连接上下圆面
        # p.addUserDebugLine(
        #     lineFromXYZ=bottom1, 
        #     lineToXYZ=top1, 
        #     lineColorRGB=[0, 1, 0],  # 绿色
        #     lineWidth=2,
        #     lifeTime=lifetime
        # )
        
        # p.addUserDebugLine(
        #     lineFromXYZ=bottom2, 
        #     lineToXYZ=top2, 
        #     lineColorRGB=[0, 1, 0],  # 绿色
        #     lineWidth=2,
        #     lifeTime=lifetime
        # )
