import numpy as np
from .gripper_hitdlr3 import Grasp, Label
from .transform import Rotation, Transform
from .simulation import *
import random
import time

T_tcp_3f_tcp=Transform(Rotation.identity(),[0,0.025,0.010])
T_tcp_2f_tcp=Transform(Rotation.identity(),[0,0.0375,0.010])

T_3f_tcp_tcp=T_tcp_3f_tcp.inverse()
T_2f_tcp_tcp=T_tcp_2f_tcp.inverse()
finger2or3_hand_width = 0.07

def direction_and_rz_select(shape_size,hand_width):
    attempt = 0
    while True:
        direction = random.choice([1,2,3,4,5,6])
        rz =[]
        # direction = 6
        if direction == 1: #下方
            if shape_size[0] < hand_width:
                rz.append(0)
                rz.append(180)
            if shape_size[1] < hand_width:
                rz.append(90)
                rz.append(270)
        elif direction == 2: #上方
            if shape_size[0] < hand_width:
                rz.append(0)
                rz.append(180)
            if shape_size[1] < hand_width:
                rz.append(90)
                rz.append(270)
        elif direction == 3: #后方
            if shape_size[0] < hand_width:
                rz.append(0)
                rz.append(180)
            if shape_size[2] < hand_width:
                rz.append(90)
                rz.append(270)
        elif direction == 4: #前方
            if shape_size[0] < hand_width:
                rz.append(0)
                rz.append(180)
            if shape_size[2] < hand_width:
                rz.append(90)
                rz.append(270)
        elif direction == 5: #左方
            if shape_size[2] < hand_width:
                rz.append(0)
                rz.append(180)
            if shape_size[1] < hand_width:
                rz.append(90)
                rz.append(270)
        elif direction == 6: #右方
            if shape_size[2] < hand_width:
                rz.append(0)
                rz.append(180)
            if shape_size[1] < hand_width:
                rz.append(90)
                rz.append(270)
        
        if len(rz) != 0:
            rz=random.choice(rz)
            return direction, rz
        attempt += 1
        if attempt >= 100:
            rz = random.choice([0,180,90,270])
            print ("No suitable direction found!")
            return direction, rz
            

def cuboid_grasp_method_stage1(sim, shape_pose, shape_size, finger_depth, hand_width): 
    if not isinstance(sim, Stage1_single_Sim):
        raise TypeError(f"Expected sim to be an instance of Stage1_single_Sim, but got {type(sim)}")
    """
    Generate grasps for cuboid objects. 
    The direction of hand opening and closing is the same as the shortest side.

    Args:
        sim: A simulation object.
        shape_pose: Pose of the object.
        shape_size: Size of the object.
        finger_depth: Finger depth of the gripper.
        hand_width: Hand width of the gripper.

    Returns:
        candidates: A list of candidate grasps.
        outcomes: A list of outcomes of the candidate grasps.
    """
    sim.restore_state()
    
    offset_rx = np.random.uniform(-20, 20)
    offset_ry = np.random.uniform(-20, 20)
    offset_rz = np.random.uniform(-20, 20)
    offset_angles = [offset_rx, offset_ry, offset_rz]
    # euler_angles = [0, 0, 0]
    # r = Rotation.from_euler('xyz',offset_angles,degrees=True)
    direction,rz = direction_and_rz_select(shape_size,hand_width)
    method = random.choice(['5finger_circle','5finger_rectangle','3finger'])
    if direction == 1: #下方
        r = Rotation.from_euler('xyz',[0,0,0],degrees=True)
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[1]
        point_offset_z = np.random.uniform(-0.35, -0.6)*shape_size[2]
    elif direction == 2: #上方
        r = Rotation.from_euler('xyz',[180,0,0],degrees=True)
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[1]
        point_offset_z = np.random.uniform(0.35, 0.6)*shape_size[2]
    elif direction == 3: #后方
        r = Rotation.from_euler('xyz',[90,0,0],degrees=True) 
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(0.35, 0.6)*shape_size[1]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[2]   
    elif direction == 4: #前方
        r = Rotation.from_euler('xyz',[-90,0,0],degrees=True) 
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(-0.35, -0.6)*shape_size[1]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[2]   
    elif direction == 5: #左方
        r = Rotation.from_euler('xyz',[0,90,0],degrees=True)
        point_offset_x = np.random.uniform(-0.35, -0.6)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[1]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[2]
    elif direction == 6: #右方
        r = Rotation.from_euler('xyz',[0,-90,0],degrees=True)
        point_offset_x = np.random.uniform(0.35, 0.6)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[1]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[2]

    rz = Rotation.from_euler('xyz',[0,0,rz],degrees=True)
    
    r = r * rz
    r = r * Rotation.from_euler('xyz',offset_angles,degrees=True)

    point_offset = [point_offset_x,point_offset_y,point_offset_z]
    point = shape_pose[0:3, 3]+point_offset
    
    pose = Transform(r, point) 
    T_world_shape = shape_pose
    T_world_grasp = pose
    if method == '3finger':
        T_grasp_tcp=T_3f_tcp_tcp
    elif method == '2finger':
        T_grasp_tcp=T_2f_tcp_tcp
    else:
        T_grasp_tcp=Transform(Rotation.identity(), [0,0,0])
    T_world_tcp = T_world_grasp*T_grasp_tcp
    T_shape_tcp = Transform.from_matrix(T_world_shape).inverse()*T_world_tcp
    
    
    pre_result,joints,exp_joints= sim.grasp_attempt(T_world_tcp, grasp_method=method)
    
    
    
    if pre_result: #二次测试
        pre_joints = joints.copy()
        pre_joints[:, 1] = joints[:, 1] * 0.6  # 指根 * 0.6
        pre_joints[:, 2] = np.minimum(joints[:, 2], joints[:, 3]) * 0.2  # 指尖*0.2
        pre_joints[:, 3] = np.minimum(joints[:, 2], joints[:, 3]) * 0.2  # 指尖*0.2
        if method == '3finger':
            pre_joints[2:4, :] = joints[2:4, :]
        if method == '2finger':
            pre_joints[1:4, :] = joints[1:4, :]
        
        sim.restore_state()
        fin_result = sim.grasp_attempt(T_world_tcp, grasp_method=method,pre_joints=pre_joints,exp_joints=exp_joints)
        
        candidate = Grasp(T_shape_tcp, pre_joints, fin_result[1], fin_result[0].value, type=method).to_grasps_dict()
        
        return candidate, fin_result[0].value
    else :
        pre_joints = np.zeros(shape=(5,4))
        candidate = Grasp(T_shape_tcp, pre_joints, joints, pre_result.value, type=method).to_grasps_dict()
        
        return candidate, pre_result.value
    
def stick_grasp_method_stage1(sim, shape_pose, shape_size, finger_depth, hand_width): 
    if not isinstance(sim, Stage1_single_Sim):
        raise TypeError(f"Expected sim to be an instance of Stage1_single_Sim, but got {type(sim)}")
    sim.restore_state()
    
    offset_rx = np.random.uniform(-20, 20)
    offset_ry = np.random.uniform(-20, 20)
    offset_rz = np.random.uniform(-20, 20)
    
    # euler_angles = [0, 0, 0]
    # r = Rotation.from_euler('xyz',offset_angles,degrees=True)
    direction,rz = direction_and_rz_select(shape_size,hand_width)
    if direction==1 or direction==2:
        method = '5finger_circle'
    elif direction==3 or direction==4:
        if shape_size[1]< 0.070: #2指或者3指的抓取宽度为0.07
            method = random.choice(['5finger_rectangle','3finger','2finger'])
        else:
            method = '5finger_rectangle'
    else:
        if shape_size[2]< 0.070:
            method = random.choice(['5finger_rectangle','3finger','2finger'])
        else:
            method = '5finger_rectangle'
    
    if direction == 1: #下方
        offset_rz = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[0,0,0],degrees=True)
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_z = np.random.uniform(-0.35, -0.6)*shape_size[1]
    elif direction == 2: #上方
        offset_rz = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[180,0,0],degrees=True)
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_z = np.random.uniform(0.35, 0.6)*shape_size[1]
    elif direction == 3: #后方
        if rz == 0 or rz == 180:
            offset_ry = np.random.uniform(-45, 45)
        else:
            offset_rx = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[90,0,0],degrees=True) 
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(0.35, 0.6)*shape_size[0]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[1]   
    elif direction == 4: #前方
        if rz == 0 or rz == 180:
            offset_ry = np.random.uniform(-45, 45)
        else:
            offset_rx = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[-90,0,0],degrees=True) 
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(-0.35, -0.6)*shape_size[0]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[1]   
    elif direction == 5: #左方
        if rz == 0 or rz == 180:
            offset_rx = np.random.uniform(-45, 45)
        else:
            offset_ry = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[0,90,0],degrees=True)
        point_offset_x = np.random.uniform(-0.35, -0.6)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[1]
    elif direction == 6: #右方
        if rz == 0 or rz == 180:
            offset_rx = np.random.uniform(-45, 45)
        else:
            offset_ry = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[0,-90,0],degrees=True)
        point_offset_x = np.random.uniform(0.35, 0.6)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[1]
        
    offset_angles = [offset_rx, offset_ry, offset_rz]
    rz = Rotation.from_euler('xyz',[0,0,rz],degrees=True)
          
    r = r * rz
    r = r * Rotation.from_euler('xyz',offset_angles,degrees=True)

    point_offset = [point_offset_x,point_offset_y,point_offset_z]
    point = shape_pose[0:3, 3]+point_offset
    
    pose = Transform(r, point) 
    T_world_shape = shape_pose
    T_world_grasp = pose
    if method == '3finger':
        T_grasp_tcp=T_3f_tcp_tcp
    elif method == '2finger':
        T_grasp_tcp=T_2f_tcp_tcp
    else:
        T_grasp_tcp=Transform(Rotation.identity(), [0,0,0])
    T_world_tcp = T_world_grasp*T_grasp_tcp
    T_shape_tcp = Transform.from_matrix(T_world_shape).inverse()*T_world_tcp
    
    pre_result,joints,exp_joints= sim.grasp_attempt(T_world_tcp, grasp_method=method)
    
    
    
    if pre_result: #二次测试
        pre_joints = joints.copy()
        pre_joints[:, 1] = joints[:, 1] * 0.6  # 指根 * 0.6
        pre_joints[:, 2] = np.minimum(joints[:, 2], joints[:, 3]) * 0.2  # 指尖*0.2
        pre_joints[:, 3] = np.minimum(joints[:, 2], joints[:, 3]) * 0.2  # 指尖*0.2
        if method == '3finger':
            pre_joints[2:4, :] = joints[2:4, :]
        if method == '2finger':
            pre_joints[1:4, :] = joints[1:4, :]
        
        sim.restore_state()
        fin_result = sim.grasp_attempt(T_world_tcp, grasp_method=method, pre_joints=pre_joints, exp_joints=exp_joints)
        
        candidate = Grasp(T_shape_tcp, pre_joints, fin_result[1], fin_result[0].value, type=method).to_grasps_dict()
        
        return candidate, fin_result[0].value
    else :
        pre_joints = np.zeros(shape=(5,4))
        candidate = Grasp(T_shape_tcp, pre_joints, joints, pre_result.value, type=method).to_grasps_dict()
        
        return candidate, pre_result.value

def cylinder_grasp_method_stage1(sim, shape_pose, shape_size, finger_depth, hand_width): 
    if not isinstance(sim, Stage1_single_Sim):
        raise TypeError(f"Expected sim to be an instance of Stage1_single_Sim, but got {type(sim)}")
    sim.restore_state()
    
    offset_rx = np.random.uniform(-20, 20)
    offset_ry = np.random.uniform(-20, 20)
    offset_rz = np.random.uniform(-20, 20)
    
    # euler_angles = [0, 0, 0]
    # r = Rotation.from_euler('xyz',offset_angles,degrees=True)
    
    direction,rz = direction_and_rz_select(shape_size,hand_width)
    if direction==1 or direction==2:
        method = '5finger_circle'
    else:
        method = '5finger_rectangle'
        
    if direction == 1: #下方
        offset_rz = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[0,0,0],degrees=True)
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_z = np.random.uniform(-0.45, -0.7)*shape_size[1]
    elif direction == 2: #上方
        offset_rz = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[180,0,0],degrees=True)
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_z = np.random.uniform(0.45, 0.7)*shape_size[1]
    elif direction == 3: #后方
        if rz == 0 or rz == 180:
            offset_ry = np.random.uniform(-45, 45)
        else:
            offset_rx = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[90,0,0],degrees=True) 
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(0.35, 0.7)*shape_size[0]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[1]   
    elif direction == 4: #前方
        if rz == 0 or rz == 180:
            offset_ry = np.random.uniform(-45, 45)
        else:
            offset_rx = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[-90,0,0],degrees=True) 
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(-0.35, -0.7)*shape_size[0]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[1]   
    elif direction == 5: #左方
        if rz == 0 or rz == 180:
            offset_rx = np.random.uniform(-45, 45)
        else:
            offset_ry = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[0,90,0],degrees=True)
        point_offset_x = np.random.uniform(-0.35, -0.7)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[1]
    elif direction == 6: #右方
        if rz == 0 or rz == 180:
            offset_rx = np.random.uniform(-45, 45)
        else:
            offset_ry = np.random.uniform(-45, 45)
        r = Rotation.from_euler('xyz',[0,-90,0],degrees=True)
        point_offset_x = np.random.uniform(0.35, 0.7)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[1]
        
    offset_angles = [offset_rx, offset_ry, offset_rz]
    rz = Rotation.from_euler('xyz',[0,0,rz],degrees=True)
          
    r = r * rz
    r = r * Rotation.from_euler('xyz',offset_angles,degrees=True)

    point_offset = [point_offset_x,point_offset_y,point_offset_z]
    point = shape_pose[0:3, 3]+point_offset
    
    pose = Transform(r, point) 
    T_world_shape = shape_pose
    T_world_grasp = pose
    if method == '3finger':
        T_grasp_tcp=T_3f_tcp_tcp
    elif method == '2finger':
        T_grasp_tcp=T_2f_tcp_tcp
    else:
        T_grasp_tcp=Transform(Rotation.identity(), [0,0,0])
    T_world_tcp = T_world_grasp*T_grasp_tcp
    T_shape_tcp = Transform.from_matrix(T_world_shape).inverse()*T_world_tcp
    
    method = random.choice(['5finger_circle','5finger_rectangle'])
    pre_result,joints,exp_joints= sim.grasp_attempt(T_world_tcp, grasp_method=method)
    
    
    if pre_result: #二次测试
        pre_joints = joints.copy()
        pre_joints[:, 1] = joints[:, 1] * 0.6  # 指根 * 0.6
        pre_joints[:, 2] = np.minimum(joints[:, 2], joints[:, 3]) * 0.2  # 指尖*0.2
        pre_joints[:, 3] = np.minimum(joints[:, 2], joints[:, 3]) * 0.2  # 指尖*0.2
        if method == '3finger':
            pre_joints[2:4, :] = joints[2:4, :]
        if method == '2finger':
            pre_joints[1:4, :] = joints[1:4, :]
        
        sim.restore_state()
        fin_result = sim.grasp_attempt(T_world_tcp, grasp_method=method,pre_joints=pre_joints,exp_joints=exp_joints)
        
        candidate = Grasp(T_shape_tcp, pre_joints, fin_result[1], fin_result[0].value, type=method).to_grasps_dict()
        
        return candidate, fin_result[0].value
    else :
        pre_joints = np.zeros(shape=(5,4))
        candidate = Grasp(T_shape_tcp, pre_joints, joints, pre_result.value, type=method).to_grasps_dict()
        
        return candidate, pre_result.value

def sphere_grasp_method_stage1(sim, shape_pose, shape_size, finger_depth, hand_width): 
    if not isinstance(sim, Stage1_single_Sim):
        raise TypeError(f"Expected sim to be an instance of Stage1_single_Sim, but got {type(sim)}")
    sim.restore_state()
    
    offset_rx = np.random.uniform(-45, 45)
    offset_ry = np.random.uniform(-45, 45)
    offset_rz = np.random.uniform(-45, 45)
    offset_angles = [offset_rx, offset_ry, offset_rz]
    # euler_angles = [0, 0, 0]
    # r = Rotation.from_euler('xyz',offset_angles,degrees=True)
    direction,rz = direction_and_rz_select(shape_size,hand_width)
    method = '5finger_circle'
    if np.mean(shape_size) < 0.075: 
        method = random.choice(['5finger_circle','5finger_rectangle','3finger'])
        
    if direction == 1: #下方
        r = Rotation.from_euler('xyz',[0,0,0],degrees=True)
        point_offset_x = np.random.uniform(-0.2, 0.2)*shape_size[0]
        point_offset_y = np.random.uniform(-0.2, 0.2)*shape_size[0]
        point_offset_z = np.random.uniform(-0.35, -0.8)*shape_size[0]
    elif direction == 2: #上方
        r = Rotation.from_euler('xyz',[180,0,0],degrees=True)
        point_offset_x = np.random.uniform(-0.2, 0.2)*shape_size[0]
        point_offset_y = np.random.uniform(-0.2, 0.2)*shape_size[0]
        point_offset_z = np.random.uniform(0.35, 0.8)*shape_size[0]
    elif direction == 3: #后方
        r = Rotation.from_euler('xyz',[90,0,0],degrees=True) 
        point_offset_x = np.random.uniform(-0.2, 0.2)*shape_size[0]
        point_offset_y = np.random.uniform(0.35, 0.8)*shape_size[0]
        point_offset_z = np.random.uniform(-0.2, 0.2)*shape_size[0]   
    elif direction == 4: #前方
        r = Rotation.from_euler('xyz',[-90,0,0],degrees=True) 
        point_offset_x = np.random.uniform(-0.2, 0.2)*shape_size[0]
        point_offset_y = np.random.uniform(-0.35, -0.8)*shape_size[0]
        point_offset_z = np.random.uniform(-0.2, 0.2)*shape_size[0]   
    elif direction == 5: #左方
        r = Rotation.from_euler('xyz',[0,90,0],degrees=True)
        point_offset_x = np.random.uniform(-0.35, -0.8)*shape_size[0]
        point_offset_y = np.random.uniform(-0.2, 0.2)*shape_size[0]
        point_offset_z = np.random.uniform(-0.2, 0.2)*shape_size[0]
    elif direction == 6: #右方
        r = Rotation.from_euler('xyz',[0,-90,0],degrees=True)
        point_offset_x = np.random.uniform(0.35, 0.8)*shape_size[0]
        point_offset_y = np.random.uniform(-0.2, 0.2)*shape_size[0]
        point_offset_z = np.random.uniform(-0.2, 0.2)*shape_size[0]
    
    rz = Rotation.from_euler('xyz',[0,0,rz],degrees=True)
          
    r = r * rz
    r = r * Rotation.from_euler('xyz',offset_angles,degrees=True)

    point_offset = [point_offset_x,point_offset_y,point_offset_z]
    point = shape_pose[0:3, 3]+point_offset
    
    pose = Transform(r, point) 
    T_world_shape = shape_pose
    T_world_grasp = pose
    if method == '3finger':
        T_grasp_tcp=T_3f_tcp_tcp
    elif method == '2finger':
        T_grasp_tcp=T_2f_tcp_tcp
    else:
        T_grasp_tcp=Transform(Rotation.identity(), [0,0,0])
    T_world_tcp = T_world_grasp*T_grasp_tcp
    T_shape_tcp = Transform.from_matrix(T_world_shape).inverse()*T_world_tcp
    
    
    pre_result,joints,exp_joints= sim.grasp_attempt(T_world_tcp, grasp_method=method)
    
    
    if pre_result: #二次测试
        pre_joints = joints.copy()
        pre_joints[:, 1] = joints[:, 1] * 0.6  # 指根 * 0.6
        pre_joints[:, 2] = np.minimum(joints[:, 2], joints[:, 3]) * 0.2  # 指尖*0.2
        pre_joints[:, 3] = np.minimum(joints[:, 2], joints[:, 3]) * 0.2  # 指尖*0.2
        if method == '3finger':
            pre_joints[2:4, :] = joints[2:4, :]
        if method == '2finger':
            pre_joints[1:4, :] = joints[1:4, :]
        
        sim.restore_state()
        fin_result = sim.grasp_attempt(T_world_tcp, grasp_method=method,pre_joints=pre_joints,exp_joints=exp_joints)
        
        candidate = Grasp(T_shape_tcp, pre_joints, fin_result[1], fin_result[0].value, type=method).to_grasps_dict()
        
        return candidate, fin_result[0].value
    else :
        pre_joints = np.zeros(shape=(5,4))
        candidate = Grasp(T_shape_tcp, pre_joints, joints, pre_result.value, type=method).to_grasps_dict()
        
        return candidate, pre_result.value

def bowl_grasp_method_stage1(sim, shape_pose, shape_size, finger_depth, hand_width): 
    if not isinstance(sim, Stage1_single_Sim):
        raise TypeError(f"Expected sim to be an instance of Stage1_single_Sim, but got {type(sim)}")
    sim.restore_state()
    
    sim.restore_state()
    
    offset_rx = np.random.uniform(-20, 20)
    offset_ry = np.random.uniform(-20, 20)
    offset_rz = np.random.uniform(-20, 20)
    
    # euler_angles = [0, 0, 0]
    # r = Rotation.from_euler('xyz',offset_angles,degrees=True)
    direction,rz = direction_and_rz_select(shape_size,hand_width)
    method = random.choice(['3finger','5finger_rectangle','2finger'])
    
    if direction == 1 or direction == 2:
        method = '5finger_circle'
    elif direction == 3 or direction == 4:
        if rz==0 or rz==180:
            offset_ry = np.random.uniform(-45, 45)
        else:
            offset_rx = np.random.uniform(-45, 45)
    else:
        if rz==0 or rz==180:
            offset_rx = np.random.uniform(-45, 45)
        else:
            offset_ry = np.random.uniform(-45, 45)
    offset_angles = [offset_rx, offset_ry, offset_rz]
    
    if direction == 1: #下方
        r = Rotation.from_euler('xyz',[0,0,0],degrees=True)
        point_offset_x = np.random.uniform(-0.3, 0.3)*shape_size[0]
        point_offset_y = np.random.uniform(-0.3, 0.3)*shape_size[1]
        point_offset_z = np.random.uniform(-0.35, -0.6)*shape_size[2]
    elif direction == 2: #上方
        r = Rotation.from_euler('xyz',[180,0,0],degrees=True)
        point_offset_x = np.random.uniform(-0.3, 0.3)*shape_size[0]
        point_offset_y = np.random.uniform(-0.3, 0.3)*shape_size[1]
        point_offset_z = np.random.uniform(0.35, 0.6)*shape_size[2]
    elif direction == 3: #后方
        r = Rotation.from_euler('xyz',[90,0,0],degrees=True) 
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(0.35, 0.6)*shape_size[1]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[2]   
    elif direction == 4: #前方
        r = Rotation.from_euler('xyz',[-90,0,0],degrees=True) 
        point_offset_x = np.random.uniform(-0.4, 0.4)*shape_size[0]
        point_offset_y = np.random.uniform(-0.35, -0.6)*shape_size[1]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[2]   
    elif direction == 5: #左方
        r = Rotation.from_euler('xyz',[0,90,0],degrees=True)
        point_offset_x = np.random.uniform(-0.35, -0.6)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[1]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[2]
    elif direction == 6: #右方
        r = Rotation.from_euler('xyz',[0,-90,0],degrees=True)
        point_offset_x = np.random.uniform(0.35, 0.6)*shape_size[0]
        point_offset_y = np.random.uniform(-0.4, 0.4)*shape_size[1]
        point_offset_z = np.random.uniform(-0.4, 0.4)*shape_size[2]

    rz = Rotation.from_euler('xyz',[0,0,rz],degrees=True)
    
    r = r * rz
    r = r * Rotation.from_euler('xyz',offset_angles,degrees=True)

    point_offset = [point_offset_x,point_offset_y,point_offset_z]
    point = shape_pose[0:3, 3]+point_offset
    
    pose = Transform(r, point) 
    T_world_shape = shape_pose
    T_world_grasp = pose
    if method == '3finger':
        T_grasp_tcp=T_3f_tcp_tcp
    elif method == '2finger':
        T_grasp_tcp=T_2f_tcp_tcp
    else:
        T_grasp_tcp=Transform(Rotation.identity(), [0,0,0])
    T_world_tcp = T_world_grasp*T_grasp_tcp
    T_shape_tcp = Transform.from_matrix(T_world_shape).inverse()*T_world_tcp
    
    
    pre_result,joints,exp_joints= sim.grasp_attempt(T_world_tcp, grasp_method=method)
    
    
    
    if pre_result: #二次测试
        pre_joints = joints.copy()
        pre_joints[:, 1] = joints[:, 1] * 0.6  # 指根 * 0.6
        pre_joints[:, 2] = np.minimum(joints[:, 2], joints[:, 3]) * 0.2  # 指尖*0.2
        pre_joints[:, 3] = np.minimum(joints[:, 2], joints[:, 3]) * 0.2  # 指尖*0.2
        if method == '3finger':
            pre_joints[2:4, :] = joints[2:4, :]
        if method == '2finger':
            pre_joints[1:4, :] = joints[1:4, :]
        
        sim.restore_state()
        fin_result = sim.grasp_attempt(T_world_tcp, grasp_method=method,pre_joints=pre_joints,exp_joints=exp_joints)
        
        candidate = Grasp(T_shape_tcp, pre_joints, fin_result[1], fin_result[0].value, type=method).to_grasps_dict()
        
        return candidate, fin_result[0].value
    else :
        pre_joints = np.zeros(shape=(5,4))
        candidate = Grasp(T_shape_tcp, pre_joints, joints, pre_result.value, type=method).to_grasps_dict()
        
        return candidate, pre_result.value

def stage2_grasp(sim, grasp_labels, obj_pose, target_id): 
    if not isinstance(sim, Stage2_multi_Sim):
        raise TypeError(f"Expected sim to be an instance of Stage1_single_Sim, but got {type(sim)}")
    num_success = 0
    num_attempt = 0
    grasp_lables_scene = []
    for grasp_label in grasp_labels:
        grasp_label['target_id']=target_id
        
        if grasp_label['label'] == 0:
            result = Label.FAILURE
            continue
        else:
            r_shape_tcp = Rotation.from_quat(grasp_label['t7d'][:4])
            T_shape_tcp = Transform(r_shape_tcp, grasp_label['t7d'][4:])
            T_world_shape = obj_pose
            T_world_tcp = T_world_shape * T_shape_tcp
            if T_world_tcp.translation[2] < 0.075:
                result = Label.FAILURE
                continue
            sim.restore_state()
            
            pre_joints = np.array(grasp_label['pre_joints']).reshape(5, 4)
            
            exprct_joints = np.array(grasp_label['exprct_joints']).reshape(5, 4)*1.15
            tip_joints = np.max(exprct_joints[:, 2:], axis=1)
            exprct_joints[:, 2:] = np.broadcast_to(tip_joints[:, np.newaxis], (5, 2))
            
            grasp_label['exprct_joints'] = exprct_joints.reshape(-1).tolist()
            
            result, joints = sim.grasp_attempt_stage2(T_world_tcp, pre_joints, exprct_joints)
            # print (result)
        if result.value == 1:
            grasp_label['label'] = 1
            num_success +=1
            num_attempt +=1
            grasp_lables_scene.append(grasp_label)
        else:
            num_attempt +=1
            grasp_label['label'] = 0
            grasp_lables_scene.append(grasp_label)
        
    sim.restore_state()      
    return num_success, grasp_lables_scene, num_attempt