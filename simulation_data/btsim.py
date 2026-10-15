import os
import time
import pickle
import numpy as np
import pybullet
from pybullet_utils import bullet_client

# from .transform import Rotation, Transform
# from .saver import get_mesh_pose_dict_from_world
from .transform import Rotation, Transform
from .saver import get_mesh_pose_dict_from_world

assert pybullet.isNumpyEnabled(), "Pybullet needs to be built with NumPy"


class BtWorld(object):
    """Interface to a PyBullet physics server.

    Attributes:
        dt: Time step of the physics simulation.
        rtf: Real time factor. If negative, the simulation is run as fast as possible.
        sim_time: Virtual time elpased since the last simulation reset.
    """

    def __init__(self, gui=True, save_dir=None, save_freq=8):
        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        self.p = bullet_client.BulletClient(connection_mode)
        
        
        # # # 截图时候用的
        self.p.configureDebugVisualizer(rgbBackground=[1,1,1])
        # # self.p.configureDebugVisualizer(shadowMapResolution=8192)
        # self.p.configureDebugVisualizer(lightPosition=[10000,10000,1000000000000])
        # self.p.configureDebugVisualizer(shadowMapIntensity=0.2)
        self.p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        # self.p.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)     # 关闭阴影
        # self.p.configureDebugVisualizer(pybullet.COV_ENABLE_LIGHTING, 0)    # 关闭光照

        # # self.p.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 1) 

        
        self.gui = gui
        self.dt = 1.0 / 240.0
        self.solver_iterations = 150
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.sim_step = 0

        self.reset()

    def set_gravity(self, gravity):
        self.p.setGravity(*gravity)

    def load_urdf(self, urdf_path, pose, scale=1.0, selfcollision=False):
        body = Body.from_urdf(self.p, urdf_path, pose, scale, SelfCollision=selfcollision)
        self.bodies[body.uid] = body
        return body
    
    def load_urdf_fixed(self, urdf_path, pose, scale=1.0):
        body = Body.from_urdf_fixed(self.p, urdf_path, pose, scale)
        self.bodies[body.uid] = body
        return body

    def remove_body(self, body):
        self.p.removeBody(body.uid)
        del self.bodies[body.uid]

    def add_constraint(self, *argv, **kwargs):
        """See `Constraint` below."""
        constraint = Constraint(self.p, *argv, **kwargs)
        return constraint

    def add_camera(self, intrinsic, near, far):
        camera = Camera(self.p, intrinsic, near, far)
        return camera

    def get_contacts(self, bodyA):
        contacts = self.p.getContactPoints(bodyA.uid)
        return contacts
    
    # def get_contacts(self, bodyA):
    #     points = self.p.getContactPoints(bodyA.uid)
    #     contacts = []
    #     for point in points:
    #         contact = Contact(
    #             bodyA=self.bodies[point[1]],
    #             bodyB=self.bodies[point[2]],
    #             linkIndexA=point[3],
    #             linkIndexB=point[4],
    #             point=point[5],
    #             normal=point[7],
    #             depth=point[8],
    #             force=point[9],
    #         )
    #         contacts.append(contact)
    #     return contacts

    def reset(self):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=self.solver_iterations
        )
        self.bodies = {}
        self.sim_time = 0.0

    def step(self):
        self.p.stepSimulation()
               
        if self.gui:
            time.sleep(self.dt)
        if self.save_dir:
            if self.sim_step % self.save_freq == 0:
                mesh_pose_dict = get_mesh_pose_dict_from_world(self, self.p._client)
                with open(os.path.join(self.save_dir, f'{self.sim_step:08d}.pkl'), 'wb') as f:
                    pickle.dump(mesh_pose_dict, f)

        self.sim_time += self.dt
        self.sim_step += 1
        
    def multi_step(self,step_num:int):
        for _ in range (step_num):
            self.step()
        

    def save_state(self):
        return self.p.saveState()

    def restore_state(self, state_uid):
        self.p.restoreState(stateId=state_uid)

    def close(self):
        self.p.disconnect()
    
    def addUserDebugLine(self,origin, end, color, lineWidth):
        self.p.addUserDebugLine(origin, end, color, lineWidth)
        
    def addUserDebugPoint(self,origin, color, size):
        self.p.addUserDebugPoint(origin, color, pointSize=size)
        
    def getDynamicsInfo(self,uid):
        # 获取惯性系的信息
        return self.p.getDynamicsInfo(uid, -1)
    
    def changeDynamics(self,uid, m, localInertia):
        self.p.changeDynamics(uid, -1, mass=m, localInertiaDiagonal=localInertia) 

    def create_constraint(self,parent_body_uid,parent_link_index,child_body_uid,child_link_index,joint_type,joint_axis):
        parent_link_state = self.p.getLinkState(parent_body_uid, parent_link_index)
        parent_link_pos = parent_link_state[0]
        parent_link_orn = parent_link_state[1]
        
        child_link_state = self.p.getLinkState(child_body_uid, child_link_index)
        child_link_pos = child_link_state[0]
        child_link_orn = child_link_state[1]
        
        constraint_id=self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_link_pos,
            parentFrameOrientation=parent_link_orn,
            childFramePosition=child_link_pos,
            childFrameOrientation=child_link_orn,
        )
        return constraint_id


class Body(object):
    """Interface to a multibody simulated in PyBullet.

    Attributes:
        uid: The unique id of the body within the physics server.
        name: The name of the body.
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
    """

    def __init__(self, physics_client, body_uid, scale):
        self.p = physics_client
        self.uid = body_uid
        self.scale = scale
        self.name = self.p.getBodyInfo(self.uid)[1].decode("utf-8")
        self.joints, self.links = {}, {}
        for i in range(self.p.getNumJoints(self.uid)):
            joint_info = self.p.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            self.joints[joint_name] = Joint(self.p, self.uid, i)
            link_name = joint_info[12].decode("utf8")
            self.links[link_name] = Link(self.p, self.uid, i)
        DynamicsInfo=self.p.getDynamicsInfo(self.uid,-1)
        inertial_pos,inertial_ori=DynamicsInfo[3],DynamicsInfo[4]
        self.T_base_inertial=Transform(Rotation.from_quat(inertial_ori),list(inertial_pos))

    @classmethod
    def from_urdf(cls, physics_client, urdf_path, pose, scale, SelfCollision=False):
        if SelfCollision:
            body_uid = physics_client.loadURDF(
                str(urdf_path),
                pose.translation,
                pose.rotation.as_quat(),
                globalScaling=scale,
                flags = physics_client.URDF_USE_SELF_COLLISION
            )
        else:
            body_uid = physics_client.loadURDF(
                str(urdf_path),
                pose.translation,
                pose.rotation.as_quat(),
                globalScaling=scale,
            )
        return cls(physics_client, body_uid, scale)

    @classmethod
    def from_urdf_fixed(cls, physics_client, urdf_path, pose, scale):
        body_uid = physics_client.loadURDF(
            str(urdf_path),
            pose.translation,
            pose.rotation.as_quat(),
            useFixedBase=True,
            globalScaling=scale,
        )
        return cls(physics_client, body_uid, scale)

    def get_pose(self):
        # 返回base的位置
        inertial_pos, inertial_ori = self.p.getBasePositionAndOrientation(self.uid)
        T_world_inertial = Transform(Rotation.from_quat(inertial_ori), np.asarray(inertial_pos))
        T_inertial_base = self.T_base_inertial.inverse()
        T_world_base = T_world_inertial*T_inertial_base
        return T_world_base
    
    def set_pose(self, T_world_base):
        # 设置base的位置
        pose = T_world_base*self.T_base_inertial
        self.p.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )
        
    def get_inertial_pose(self):
        # 返回世界坐标系下惯性系坐标系的位置
        pos, ori = self.p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), np.asarray(pos))
    
    def set_inertial_pose(self, pose):
        # 设置惯性系的位置
        self.p.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )

    def get_velocity(self):
        linear, angular = self.p.getBaseVelocity(self.uid)
        return linear, angular


class Link(object):
    """Interface to a link simulated in Pybullet.

    Attributes:
        link_index: The index of the joint.
    """

    def __init__(self, physics_client, body_uid, link_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.link_index = link_index

    def get_pose(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos, ori = link_state[0], link_state[1]
        return Transform(Rotation.from_quat(ori), pos)


class Joint(object):
    """Interface to a joint simulated in PyBullet.

    Attributes:
        joint_index: The index of the joint.
        lower_limit: Lower position limit of the joint.
        upper_limit: Upper position limit of the joint.
        effort: The maximum joint effort.
    """

    def __init__(self, physics_client, body_uid, joint_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.joint_index = joint_index

        joint_info = self.p.getJointInfo(body_uid, joint_index)
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.effort = joint_info[10]

    def get_position(self):
        joint_state = self.p.getJointState(self.body_uid, self.joint_index)
        return joint_state[0]

    def set_position(self, position, kinematics=False):
        if kinematics:
            self.p.resetJointState(self.body_uid, self.joint_index, position)
        self.p.setJointMotorControl2(
            self.body_uid,
            self.joint_index,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.effort,
        )


class Constraint(object):
    """Interface to a constraint in PyBullet.

    Attributes:
        uid: The unique id of the constraint within the physics server.
    """

    def __init__(
        self,
        physics_client,
        parent,
        parent_link,
        child,
        child_link,
        joint_type,
        joint_axis,
        parent_frame,
        child_frame,
    ):
        """
        Create a new constraint between links of bodies.

        Args:
            parent:
            parent_link: None for the base.
            child: None for a fixed frame in world coordinates.

        """
        self.p = physics_client
        parent_body_uid = parent.uid
        parent_link_index = parent_link.link_index if parent_link else -1
        child_body_uid = child.uid if child else -1
        child_link_index = child_link.link_index if child_link else -1

        self.uid = self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_frame.translation,
            parentFrameOrientation=parent_frame.rotation.as_quat(),
            childFramePosition=child_frame.translation,
            childFrameOrientation=child_frame.rotation.as_quat(),
        )

    def change(self, **kwargs):
        self.p.changeConstraint(self.uid, **kwargs)


class Contact(object):
    """Contact point between two multibodies.

    Attributes:
        point: Contact point.
        normal: Normal vector from ... to ...
        depth: Penetration depth
        force: Contact force acting on body ...
    """

    def __init__(self, bodyA, bodyB, linkIndexA, linkIndexB, point, normal, depth, force):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.linkIndexA = linkIndexA
        self.linkIndexB = linkIndexB
        self.point = point
        self.normal = normal
        self.depth = depth
        self.force = force


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, physics_client, intrinsic, near, far):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.p = physics_client

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )

        rgb, z_buffer = result[2][:, :, :3], result[3]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )
        seg = result[4] # by xgh
        return rgb, depth, seg

def render_with_keypoints(self, extrinsic, keypoints):
    """Render synthetic RGB and depth images.

    Args:
        extrinsic: Extrinsic parameters, T_cam_ref.
        keypoints: A list of keypoints in the reference frame.  
    Returns:
        rgb: RGB image.
        depth: Depth image.
        projected_keypoints: A list of keypoints projected onto the image plane.
    """
    gl_view_matrix = extrinsic.as_matrix()
    gl_view_matrix[2, :] *= -1  # flip the Z axis
    gl_view_matrix = gl_view_matrix.flatten(order="F")
    gl_proj_matrix = self.proj_matrix.flatten(order="F")

    result = self.p.getCameraImage(
        width=self.intrinsic.width,
        height=self.intrinsic.height,
        viewMatrix=gl_view_matrix,
        projectionMatrix=gl_proj_matrix,
        renderer=pybullet.ER_TINY_RENDERER,
    )

    rgb, z_buffer = result[2][:, :, :3], result[3]
    depth = (
        1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
    )

    projected_keypoints = []
    for keypoint in keypoints:
        # Check if the keypoint is within the viewing frustum
        keypoint_homogeneous = np.append(keypoint, [1])
        keypoint_view = gl_view_matrix.dot(keypoint_homogeneous)
        if keypoint_view[2] < 0:
            continue

        # Project the keypoint onto the image plane
        keypoint_screen = gl_proj_matrix.dot(keypoint_view)
        keypoint_screen /= keypoint_screen[3]
        keypoint_screen = keypoint_screen[:3]

        # Check if the keypoint is visible in the current view
        if (
            keypoint_screen[0] < -1 or
            keypoint_screen[0] > 1 or
            keypoint_screen[1] < -1 or
            keypoint_screen[1] > 1 or
            keypoint_screen[2] < 0
        ):
            continue

        # Convert normalized coordinates to pixel coordinates
        pixel_x = int((keypoint_screen[0] + 1) * (self.intrinsic.width - 1) / 2)
        pixel_y = int((1 - keypoint_screen[1]) * (self.intrinsic.height - 1) / 2)

        # Check if the keypoint is occluded by other objects
        keypoint_depth = depth[pixel_y, pixel_x]
        if keypoint_view[2] > keypoint_depth:
            continue

        projected_keypoints.append((pixel_x, pixel_y))

    return rgb, depth, projected_keypoints

def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho
