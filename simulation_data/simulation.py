from pathlib import Path
import time
import math
import numpy as np
import pybullet as p
import random

from . import btsim
from .transform import Rotation, Transform
from .data_utils import CameraIntrinsic
from .gripper_hitdlr3 import *
from .saver import *

class Stage1_single_Sim(object):
    def __init__(self, object_set=None, gui=True, seed=None, urdf_root = "./data/urdfs"):
        
        self.urdf_root = Path(urdf_root)
        self.object_set = object_set
        # self.discover_objects()

        self.global_scaling = {"blocks": 1.0}.get(object_set, 1.0)
        self.gui = gui

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui)
        self.gripper = Gripper(self.world)
        # intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)  # vgn
        intrinsic = CameraIntrinsic(640, 480, 606.73, 606.35, 321.017, 254.109)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)
        self.success_grasp_count = 0
        self.grasp_try_count = 0
    
    def reset(self):
        self.world.reset()
        # self.world.set_gravity([0.0, 0.0, -9.81])
        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=0.3,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.0, -0.1, 1.1],
            )             
    
    def discover_objects(self):
        root = self.urdf_root / self.object_set
        self.object_urdfs = sorted(f for f in root.iterdir() if f.suffix == ".urdf")
        self.num_urdf = len(self.object_urdfs)
        return self.num_urdf
    
    def save_state(self):
        self._snapshot_id = self.world.save_state()
    
    def restore_state(self):
        self.world.restore_state(self._snapshot_id)
        
    def fixed_body(self):
        self.world.changeDynamics(self.body_id, 0, [0,0,0])
        
    def defixed_body(self):
        self.world.changeDynamics(self.body_id, self.body_mess, self.body_localInertia)
    
    def load_urdf(self, object_set, name, scale):
        self.object_set = object_set
        self.save_state()
        root = self.urdf_root / self.object_set
        self.object_urdfs = sorted(f for f in root.iterdir() if f.suffix == ".urdf")
        
        for file_path in self.object_urdfs:
            if file_path.name.endswith(name+".urdf"):
                urdf = file_path
                break
        else:
            raise FileNotFoundError(f"未在 {self.object_set} 类中找到 {name} 模型文件")

        x = 0.0
        y = 0.0
        z = 1.0
        angle = 0
        rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 0.0]) #* Rotation.from_rotvec(np.pi/2 * np.r_[0.0, 1.0, 0.0])
        pre_pose = Transform(rotation, np.r_[x, y, z])
        body = self.world.load_urdf(urdf, pre_pose, scale=self.global_scaling * scale)
        body_info = self.world.getDynamicsInfo(body.uid)
        self.body_id = body.uid
        self.body_mess = body_info[0]
        self.body_localInertia = body_info[2]
        self.fixed_body()
        ids = body.uid
        self.world.step()

        return ids, urdf

    def load_gripper(self, T_world_tcp, joints = None):
        # self.save_state()
        self.fixed_body()
        self.gripper.reset(T_world_tcp)
        if joints == None:
            gripper_joints = [[0,0.35,0,0],
                                [0,0.12,0,0],
                                [0,-0.12,0,0],
                                [0,-0.35,0,0],
                                [0,-0.12,0,0]]
        else:
            gripper_joints = [[joints[0],joints[1],joints[2],joints[3]],
                                [joints[4],joints[5],joints[6],joints[7]],
                                [joints[8],joints[9],joints[10],joints[11]],
                                [joints[12],joints[13],joints[14],joints[15]],
                                [joints[16],joints[17],joints[18],joints[19]]]
        self.gripper.set_joint_pose(gripper_joints)
        self.world.step()
        
    def remove_gripper(self):
        self.world.remove_body(self.gripper.body)

    def gengerate_stage1_scence(self, urdf_idx, scale):

        self.save_state()
        urdf = self.object_urdfs[urdf_idx]
        # scale = np.random.uniform(0.7, 1.0)
        x = 0.0
        y = 0.0
        z = 1.0
        angle = 0
        rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 0.0]) #* Rotation.from_rotvec(np.pi/2 * np.r_[0.0, 1.0, 0.0])

        pre_pose = Transform(rotation, np.r_[x, y, z])
        body = self.world.load_urdf(urdf, pre_pose, scale=self.global_scaling * scale)
        body_info = self.world.getDynamicsInfo(body.uid)
        self.body_id = body.uid
        self.body_mess = body_info[0]
        self.body_localInertia = body_info[2]
        self.fixed_body()
        ids = body.uid
        
        lower, upper = self.world.p.getAABB(body.uid) #bounding box
        size_x = upper[0]-lower[0]
        size_y = upper[1]-lower[1]
        size_z = upper[2]-lower[2]
        
        #shape_size, shape
        # if self.object_set == 'bowl': 
        #     h = size_z
        #     r = size_x / 2 
        #     shape_size = np.array([r, h])
        #     shape = 'bowl'
        # elif self.object_set == 'cylinder' : 
        #     h = size_z
        #     r = size_x / 2 
        #     shape_size = np.array([r, h])
        #     shape = 'cylinder'
        # elif self.object_set == 'stick': 
        #     h = size_z
        #     r = size_x / 2 
        #     shape_size = np.array([r, h])
        #     shape = 'stick'
        # elif self.object_set == 'ring': # ring
        #     h = size_z
        #     r = size_x / 2
        #     shape_size = np.array([r, h])
        #     shape = 'ring'
        # elif self.object_set == 'sphere': # sphere
        #     r = size_x / 2
        #     shape_size = np.array([r,])
        #     shape = 'sphere'
        # elif self.object_set == 'cuboid'or self.object_set == '1': # cuboid
        #     shape_size = np.array([size_x, size_y, size_z])
        #     shape = 'cuboid'
        
        shape = self.object_set
        shape_scale = scale
        shape_size = np.array([size_x, size_y, size_z])

        #pose
        # pos, ori = self.world.p.getBasePositionAndOrientation(body.uid)
        # ori = self.world.p.getMatrixFromQuaternion(ori)
        # inf = self.world.p.getVisualShapeData(body.uid)
        
        # pose_m = np.eye(4, 4)
        # pose_m[:3, 3] = pos
        # pose_m[0, :3] = ori[0:3]
        # pose_m[1, :3] = ori[3:6]
        # pose_m[2, :3] = ori[6:9]
        # pose_m[-1, -1] = 1
        # pose = pose_m
        self.world.step()
        T_world_shape = body.get_pose()
        pose = T_world_shape.as_matrix()
          
        return ids, urdf, shape, pose, shape_size,shape_scale
    
    def grasp_attempt(self, T_world_tcp, grasp_method='5finger_circle', pre_joints=None, exp_joints=None, allow_contact=False):
        self.fixed_body()
        T_world_grasp = T_world_tcp 
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.15])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        self.gripper.reset(T_world_pregrasp)
        
        if pre_joints is not None:
            self.gripper.set_joint_pose(pre_joints)
        else:
            self.gripper.random_pre_grasp_pose(grasp_method)
        

        self.grasp_try_count += 1

        if self.gripper.detect_contact()[0]:
            result = Label.FAILURE, self.gripper.read_joint_pose()
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            
            if self.gripper.detect_contact()[0] and not allow_contact:
                result = Label.FAILURE, self.gripper.read_joint_pose()
            else:
                if exp_joints is not None:
                    self.gripper.graspit(joints=exp_joints)
                else:
                    exp_joints=self.gripper.graspit(grasp_method)
                
                self.defixed_body()
                
                self.world.set_gravity([0.0, 0.0, -9.81])
                self.world.multi_step(10)
                if self.check_success(self.gripper):
                    self.world.set_gravity([0.0, 0.0, 9.81])
                    self.world.multi_step(10)
                    if self.check_success(self.gripper):
                        self.world.set_gravity([0.0, -9.81, 0.0])
                        self.world.multi_step(10)
                        if self.check_success(self.gripper):
                            self.world.set_gravity([0.0, 9.81, 0.0])
                            self.world.multi_step(10)  
                            if self.check_success(self.gripper):
                                self.world.set_gravity([-9.81, 0.0, 0.0])
                                self.world.multi_step(10)
                                if self.check_success(self.gripper):
                                    self.world.set_gravity([9.81, 0.0, 0.0])
                                    self.world.multi_step(10)
                                    if self.check_success(self.gripper):                                                               
                                        result = Label.SUCCESS, self.gripper.read_joint_pose()
                                    else:
                                        result = Label.FAILURE, self.gripper.read_joint_pose()
                                else:
                                    result = Label.FAILURE, self.gripper.read_joint_pose()
                            else:
                                result = Label.FAILURE, self.gripper.read_joint_pose()
                        else:
                            result = Label.FAILURE, self.gripper.read_joint_pose()
                    else:
                        result = Label.FAILURE, self.gripper.read_joint_pose() 
                else:
                    result = Label.FAILURE, self.gripper.read_joint_pose()
            self.world.set_gravity([0.0, 0.0, 0.0])
        self.world.remove_body(self.gripper.body)
        return *result,exp_joints
        # print (self.success_grasp_count)    
    
    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts,contact_points = gripper.detect_contact()
        current_angles = gripper.read_joint_pose()
        min_mask=(gripper.joints_min_limits.reshape(5, 6)[:, 2:-1] * 0.95) < current_angles[:,1:]
        max_mask=current_angles[:,1:] < (gripper.joints_max_limits.reshape(5, 6)[:, 2:-1] * 0.96)
        is_within_limits = np.all(min_mask) and np.all(max_mask)
        res = contacts and is_within_limits
        return res
    
    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()
                
    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break    

def workspace_lines(size):
    return [
        [-size/2, size/2, 0.0],
        [size/2, size/2, 0.0],
        [size/2, size/2, 0.0],
        [size/2, -size/2, 0.0],
        [size/2, -size/2, 0.0],
        [-size/2, -size/2, 0.0],
        [-size/2, -size/2, 0.0],
        [-size/2, size/2, 0.0],
        [-size/2, size/2, 0.0],
        [-size/2, size/2, size],
        [size/2, size/2, 0.0],
        [size/2, size/2, size],
        [size/2, -size/2, 0.0],
        [size/2, -size/2, size],
        [-size/2, -size/2, 0.0],
        [-size/2, -size/2, size],
        [-size/2, size/2, size],
        [size/2, size/2, size],
        [size/2, size/2, size],
        [size/2, -size/2, size],
        [size/2, -size/2, size],
        [-size/2, -size/2, size],
        [-size/2, -size/2, size],
        [-size/2, size/2, size],
    ]
                
class Stage2_multi_Sim(object):
    def __init__(self, gui=True, data_root ="./data", urdf_root = "./data/urdfs",random = None):
        
        self.urdf_root = Path(urdf_root)
        # self.discover_objects()
        self.table_height = 0.05
        self.gui = gui
        self.data_root = data_root

        self.rng = random if random else np.random
        
        self.world = btsim.BtWorld(self.gui)
        self.gripper = Gripper(self.world)
        self.workspace_size = 0.6
        intrinsic = CameraIntrinsic(640, 480, 525.0, 525.0, 320.0, 240.0) #仿真环境
        # intrinsic = CameraIntrinsic(640, 480, 606.73, 606.35, 321.017, 254.109)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)
        self.success_grasp_count = 0
    
    def reset(self):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        # self.draw_workspace()
        # if self.gui:
        #     self.world.p.resetDebugVisualizerCamera(
        #         cameraDistance=0.3,
        #         cameraYaw=0.0,
        #         cameraPitch=-45,
        #         cameraTargetPosition=[0.0, -0.1, 1.1],
        #     )
        # self.place_plane(self.table_height)
        # self.place_shelves(self.table_height)
        # self.place_desk(self.table_height)
        # self.place_table(self.table_height)
        # self.place_table1(self.table_height)
        # self.place_table2(self.table_height)
        # self.place_shelves1(self.table_height)
        
        placement_methods = [
            self.place_plane,
            # self.place_shelves,
            # self.place_desk,
            # self.place_table,
            # self.place_table1,
            # self.place_table2,
            # self.place_shelves1
        ]
        # 随机选择一个执行
        selected_method = self.rng.choice(placement_methods)
        selected_method(self.table_height)
    
    def place_shelves(self, height):
        file_path=self.urdf_root / "setup" / "shelves" / "shelves.obj"
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=str(file_path),
            collisionFramePosition=[0, 0, 0],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH  # 确保是凹形
        )
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(file_path),
            visualFramePosition=[0, 0, 0]
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, height-0.011]
        )
    
    def place_shelves1(self, height):
        file_path=self.urdf_root / "setup" / "shelves1" / "shelves1.obj"
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=str(file_path),
            collisionFramePosition=[0, 0, 0],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH  # 确保是凹形
        )
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(file_path),
            visualFramePosition=[0, 0, 0]
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, height]
        )
    
    def place_desk(self, height):
        urdf = self.urdf_root / "setup" / "desk.urdf"
        pose = Transform(Rotation.identity(), [0, 0, height+0.013])
        plane = self.world.load_urdf(urdf, pose, scale=1)
    
    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "table.urdf"
        pose = Transform(Rotation.identity(), [0, 0, height])
        plane = self.world.load_urdf(urdf, pose, scale=1)
    
    def place_table1(self, height):
        urdf = self.urdf_root / "setup" / "table1.urdf"
        pose = Transform(Rotation.identity(), [0, 0, height])
        plane = self.world.load_urdf(urdf, pose, scale=1)
        
    def place_table2(self, height):
        urdf = self.urdf_root / "setup" / "table2.urdf"
        pose = Transform(Rotation.identity(), [0, 0, height+0.0002])
        plane = self.world.load_urdf(urdf, pose, scale=1)
    
    def place_plane(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0, 0, height])
        plane = self.world.load_urdf(urdf, pose, scale=1)
        p.changeVisualShape(plane.uid, -1, rgbaColor=[0.9, 0.9, 0.9, 1])
     
    def discover_objects(self, object_sets):
        urdf = []
        object_class = []
        self.object_set_list = object_sets
        for object_set in self.object_set_list:
            path = self.urdf_root/object_set
            for f in sorted(path.iterdir()):
                if f.suffix == ".urdf":
                    urdf.append(f)
                    object_class.append(object_set)
        self.urdf = {
            'object_set':object_class,
            'urdf':urdf
        }
        self.num_urdf = len(urdf)
        return self.num_urdf
    
    def save_state(self):
        self._snapshot_id = self.world.save_state()
    
    def restore_state(self):
        self.world.restore_state(self._snapshot_id)
        
    def fixed_body(self):
        self.world.changeDynamics(self.body_id, 0, [0,0,0])
        
    def defixed_body(self):
        self.world.changeDynamics(self.body_id, self.body_mess, self.body_localInertia)
    
    def load_urdf(self, name, obj_scale, pose):
        pose = Transform.from_list(pose)
        file_path = name
        try:
            body = self.world.load_urdf(file_path, pose, scale= obj_scale)
        except Exception as e:
            print(f"读取失败：{e}")
        body_info = self.world.getDynamicsInfo(body.uid)
        self.body_id = body.uid
        self.body_mess = body_info[0]
        self.body_localInertia = body_info[2]
        ids = body.uid
        self.world.step()
        self.save_state()

        return ids

    def load_gripper(self, T_world_tcp, joints):
        # self.save_state()
        # self.fixed_body()
        self.gripper.reset(T_world_tcp)
        gripper_joints = [[joints[0],joints[1],joints[2],joints[3]],
                            [joints[4],joints[5],joints[6],joints[7]],
                            [joints[8],joints[9],joints[10],joints[11]],
                            [joints[12],joints[13],joints[14],joints[15]],
                            [joints[16],joints[17],joints[18],joints[19]]]
        self.gripper.set_joint_pose(gripper_joints)
        self.world.step()
        
    def remove_gripper(self):
        self.world.remove_body(self.gripper.body)

    def draw_workspace(self):
        points = workspace_lines(self.workspace_size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )
            
    def random_pose(self, shuffle = False):
        if not shuffle:
            x = self.rng.uniform(-0.3, 0.3)
            y = self.rng.uniform(-0.3, 0.3)
            z = self.rng.uniform(0.01, 0.05)
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0]) 
            pose = Transform(rotation, np.r_[x, y, z])
        else:
            x = self.rng.uniform(-0.2, 0.2)
            y = self.rng.uniform(-0.2, 0.2)
            z = self.rng.uniform(0.01, 0.15)
            angle_x = self.rng.uniform(-0.5 * np.pi, 0.5 * np.pi)
            angle_y = self.rng.uniform(-0.5 * np.pi, 0.5 * np.pi)
            angle_z = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec([angle_x, angle_y, angle_z])
            pose = Transform(rotation, np.r_[x, y, z])
        return pose
            
    def gengerate_stage2_scence(self, scence_obj_num, scales, min_grasps_per_urdf, shuffle = False):
        urdf_idxs = self.rng.choice(self.num_urdf, scence_obj_num, replace=False)
        self.bodys_id = []
        self.urdf_name = []
        self.object_set = []
        self.scale = []
        self.bodys =[]
        self.bodys_pose = []
        max_attempts = 10
        
        for urdf_idx in urdf_idxs:
            attempts = 0
            self.save_state()
            urdf_name = self.urdf['urdf'][urdf_idx]
            object_set = self.urdf['object_set'][urdf_idx]
            whether_reserve = False
            self.rng.shuffle(scales)
            for scale in scales:
                filename = find_grasp_lables_file(self.data_root, urdf_name, object_set, scale)
                stage1_num_success = read_grasp_labels_stage1(filename, only_read_success=True)
                if stage1_num_success is not None and stage1_num_success >= min_grasps_per_urdf:
                    whether_reserve = True
                    break
            if not whether_reserve:
                continue
            
            pre_pose = self.random_pose(shuffle)
            
            body = self.world.load_urdf(urdf_name, pre_pose, scale=scale )
            p.changeVisualShape(body.uid, -1, rgbaColor=[1, 0.5, 0, 1])
            lower, upper = self.world.p.getAABB(body.uid)
            z = self.table_height + 0.5 * (upper[2] - lower[2]) + pre_pose.translation[2]
            pre_pose.translation[2] = z
            body.set_pose(pre_pose)
            self.world.step()
            
            self.bodys_id.append(body.uid)
            self.urdf_name.append(urdf_name)
            self.object_set.append(object_set)
            self.scale.append(scale)
            self.bodys.append(body)
            
            while self.world.get_contacts(body) and attempts < max_attempts:
                re_pose = self.random_pose(shuffle)
                z = self.table_height + 0.5 * (upper[2] - lower[2]) + re_pose.translation[2]
                re_pose.translation[2] = z
                body.set_pose(re_pose)
                self.world.step()
                attempts += 1
            if self.world.get_contacts(body):
                self.world.remove_body(body)
                del self.bodys_id[-1] 
                del self.urdf_name[-1]
                del self.object_set[-1]
                del self.scale[-1]
                del self.bodys[-1]

        self.wait_for_objects_to_rest(timeout=2.0)
        self.remove_and_wait()
        self.save_state()
        self.restore_state()
        for body in self.bodys:
            try:
                T_world_body = body.get_pose()
            except:
                pass
                # print("such object has been removed")
            else:
                self.bodys_pose.append(T_world_body)

        return self.bodys_id, self.urdf_name, self.object_set, self.scale, self.bodys_pose
    
    def render_image(self,radius,theta,phi):
        origin = Transform(Rotation.identity(), np.r_[0, 0, 0])
        extrinsic = self.camera_on_sphere(origin, radius, theta, phi)
        rgb,depth,seg = self.camera.render(extrinsic)
        return rgb, depth, seg, extrinsic
    
    def camera_on_sphere(self, origin, radius, theta, phi, target=np.array([0.0, 0.0, 0.0])):
        eye = np.r_[
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ] + target
        up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
        return Transform.look_at(eye, target, up) * origin.inverse()  # T_eye_ref*T_ref_world
     
    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()
            
    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if xyz[2] < 0.0 or np.any(abs(xyz) > self.workspace_size/2):
                # print (body.uid)
                self.world.remove_body(body)
                del_index = self.bodys_id.index(body.uid)
                del self.bodys_id[del_index] 
                del self.urdf_name[del_index]
                del self.object_set[del_index]
                del self.scale[del_index]
                removed_object = True
        return removed_object
                
    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break      
    
    def grasp_attempt_stage2(self, T_world_grasp, pre_joints, exprct_joints, allow_contact=False):
        
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.10])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        self.gripper.reset(T_world_pregrasp)
        pre_joints = np.array(pre_joints).reshape(5, 4)
        self.gripper.set_joint_pose(pre_joints)

        if self.gripper.detect_contact()[0]:
            result = Label.FAILURE, self.gripper.read_joint_pose()
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact()[0] and not allow_contact:
                result = Label.FAILURE, self.gripper.read_joint_pose()
            else:

                self.gripper.graspit(joints = exprct_joints)
                # 提升物体
                # T_world_lift = Transform(Rotation.identity(), [0.0, 0.0, 0.2]) * T_world_grasp 
                # self.gripper.move_tcp_xyz(T_world_lift, abort_on_contact=False)
                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read_joint_pose()
                else:
                    result = Label.FAILURE, self.gripper.read_joint_pose()
        self.world.remove_body(self.gripper.body)
        return result
    
    def grasp_disturbance_test(self, T_world_grasp, pre_joints, exprct_joints, allow_contact=False):
        
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.10])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        self.gripper.reset(T_world_pregrasp)
        pre_joints = np.array(pre_joints).reshape(5, 4)
        self.gripper.set_joint_pose(pre_joints)

        if self.gripper.detect_contact()[0]:
            result = Label.FAILURE, self.gripper.read_joint_pose()
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact()[0] and not allow_contact:
                result = Label.FAILURE, self.gripper.read_joint_pose()
            else:

                self.gripper.graspit(joints = exprct_joints)
                # 提升物体
                # T_world_lift = Transform(Rotation.identity(), [0.0, 0.0, 0.2]) * T_world_grasp 
                # self.gripper.move_tcp_xyz(T_world_lift, abort_on_contact=False)
                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read_joint_pose()
                else:
                    result = Label.FAILURE, self.gripper.read_joint_pose()
        self.world.remove_body(self.gripper.body)
        return result
    
    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts,contact_points = gripper.detect_contact()
        current_angles = gripper.read_joint_pose()
        min_mask=(gripper.joints_min_limits.reshape(5, 6)[:, 2:-1] * 0.95) < current_angles[:,1:]
        max_mask=current_angles[:,1:] < (gripper.joints_max_limits.reshape(5, 6)[:, 2:-1] * 0.96)
        is_within_limits = np.all(min_mask) and np.all(max_mask)
        res = contacts and is_within_limits
        return res
