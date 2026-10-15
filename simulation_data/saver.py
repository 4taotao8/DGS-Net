import pickle
import pybullet
import numpy as np
import os
from pathlib import Path
import cv2
# import open3d as o3d
# import PIL.Image as Image
from .transform import Rotation, Transform

#  "/home/gt/gt/DGO/gt_graspdata/data"
# CUR_PATH = os.path.dirname(os.path.abspath(__file__))
# DATA_ROOT =os.path.join(CUR_PATH, '../data')
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get('DGS_DATA_ROOT', os.path.join(CUR_PATH, '../data'))

def get_mesh_pose_dict_from_world(world,
                                  physicsClientId=0,
                                  exclude_plane=True):
    mesh_pose_dict = {}
    for uid in world.bodies.keys():
        _, name = world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' and exclude_plane:
            continue
        body = world.bodies[uid]
        for visual in world.p.getVisualShapeData(uid, physicsClientId):
            object_name, mesh_path, scale, pose = get_mesh_pose(
                visual, physicsClientId)
            mesh_path = mesh_path.decode('utf8')
            mesh_pose_dict[object_name] = (mesh_path, scale, pose)
    return mesh_pose_dict

def get_mesh_pose(visual, physicsClientId):
    objectUniqueId = visual[0]
    linkIndex = visual[1]
    visualGeometryType = visual[2]
    scale = visual[3]
    meshAssetFileName = visual[4]
    localVisualFramePosition = visual[5]
    localVisualFrameOrientation = visual[6]
    rgbaColor = visual[7]

    visual_offset = Transform(Rotation.from_quat(localVisualFrameOrientation),
                              np.array(localVisualFramePosition))
    if linkIndex != -1:
        linkState = get_link_pose((objectUniqueId, linkIndex), physicsClientId)
        linkOffsetState = get_link_local_offset((objectUniqueId, linkIndex), physicsClientId)
        linkOffsetState.translation = np.array([0, 0, 0])
        linkOffsetState = linkOffsetState * visual_offset
    else:
        linkState = get_body_pose(objectUniqueId, physicsClientId)
        linkOffsetState = visual_offset
#     transform = linkState
    transform = linkState * linkOffsetState
    
    # Name to use for visii components
    object_name = str(objectUniqueId) + "_" + str(linkIndex)
    return object_name, meshAssetFileName, scale, transform

def get_link_local_offset(link_uid, physicsClientId=0):
    """Get the local offset of the link.
    Args:
        link_uid: A tuple of the body Unique ID and the link index.
    Returns:
        An instance of Pose.
    """
    body_uid, link_ind = link_uid
    _, _, position, quaternion, _, _ = pybullet.getLinkState(
        bodyUniqueId=body_uid, linkIndex=link_ind, physicsClientId=physicsClientId)
    ori = Rotation.from_quat(quaternion)
    return Transform(ori, np.array(position))

def get_link_pose(link_uid, physicsClientId):
    """Get the pose of the link.
    Args:
        link_uid: A tuple of the body Unique ID and the link index.
    Returns:
        An instance of Pose.
    """
    body_uid, link_ind = link_uid
    _, _, _, _, position, quaternion = pybullet.getLinkState(
        bodyUniqueId=body_uid, linkIndex=link_ind,
        physicsClientId=physicsClientId)
    ori = Rotation.from_quat(quaternion)
    return Transform(ori, np.array(position))

def get_link_center_pose(link_uid, physicsClientId):
    """Get the pose of the link center of mass.
    Args:
        link_uid: A tuple of the body Unique ID and the link index.
    Returns:
        An instance of Pose.
    """
    body_uid, link_ind = link_uid
    position, quaternion, _, _, _, _ = pybullet.getLinkState(
        bodyUniqueId=body_uid, linkIndex=link_ind,
        physicsClientId=physicsClientId)
    ori = Rotation.from_quat(quaternion)
    return Transform(ori, np.array(position))

def get_body_pose(body_uid, physicsClientId):
    """Get the pose of the body.
    The pose of the body is defined as the pose of the base of the body.
    Args:
        body_uid: The body Unique ID.
    Returns:
        An instance of Pose.
    """
    position, quaternion = pybullet.getBasePositionAndOrientation(
        bodyUniqueId=body_uid, physicsClientId=physicsClientId)

    tmp = pybullet.getDynamicsInfo(body_uid, -1, physicsClientId)
    local_inertial_pos, local_inertial_ori = tmp[3], tmp[4]
    local_transform = Transform(Rotation.from_quat(local_inertial_ori),
                                np.array(local_inertial_pos))

    base_frame_transform = Transform(
        Rotation.from_quat(quaternion),
        np.array(position)) * local_transform.inverse()

    return base_frame_transform

def save_grasp_lables(object_set, name, scale, grasp_labels, num_success):
    try:
        # 打开文件，如果不存在则创建
        name = os.path.basename(name).split('.')[0]
        
        label_file = os.path.join(DATA_ROOT,f"grasp_labels/{object_set}/{name}")
        if not os.path.exists(label_file):
            os.makedirs(label_file)
        filename = os.path.join(label_file, f"{name}_{scale}.txt")
        
        with open(filename , 'w') as file:
            # 遍历每个抓取标签
            line1 = f"object_set={object_set}    name={name}    scale={scale}    num_success={num_success}\n"
            file.write(line1)
            line2 = f"qx  qy  qz  qw  x  y  z  pre_index0  pre_index1  pre_index2  pre_index3  "\
                    f"pre_mid0  pre_mid1  pre_mid2  pre_mid3  pre_ring0  pre_ring1  pre_ring2  pre_ring3  "\
                    f"pre_little0  pre_little1  pre_little2  pre_little3  pre_thumb0  pre_thumb1  pre_thumb2  pre_thumb3  "\
                    f"index0  index1  index2  index3  mid0  mid1  mid2  mid3  "\
                    f"ring0  ring1  ring2  ring3  little0  little1  little2  little3  "\
                    f"thumb0  thumb1  thumb2  thumb3  label  type (0_cebai 1_zhigeng 2_zhijian 3_zhijian)\n"
            file.write(line2)
            for data in grasp_labels:
                # 将数据转换为一行字符串
                line =  f"{data['t7d'][0]} {data['t7d'][1]} {data['t7d'][2]} {data['t7d'][3]} {data['t7d'][4]} {data['t7d'][5]} {data['t7d'][6]} "\
                        f"{data['pre_joints'][0]} {data['pre_joints'][1]} {data['pre_joints'][2]} {data['pre_joints'][3]} "\
                        f"{data['pre_joints'][4]} {data['pre_joints'][5]} {data['pre_joints'][6]} {data['pre_joints'][7]} "\
                        f"{data['pre_joints'][8]} {data['pre_joints'][9]} {data['pre_joints'][10]} {data['pre_joints'][11]} "\
                        f"{data['pre_joints'][12]} {data['pre_joints'][13]} {data['pre_joints'][14]} {data['pre_joints'][15]} "\
                        f"{data['pre_joints'][16]} {data['pre_joints'][17]} {data['pre_joints'][18]} {data['pre_joints'][19]} "\
                        f"{data['exprct_joints'][0]} {data['exprct_joints'][1]} {data['exprct_joints'][2]} {data['exprct_joints'][3]} "\
                        f"{data['exprct_joints'][4]} {data['exprct_joints'][5]} {data['exprct_joints'][6]} {data['exprct_joints'][7]} "\
                        f"{data['exprct_joints'][8]} {data['exprct_joints'][9]} {data['exprct_joints'][10]} {data['exprct_joints'][11]} "\
                        f"{data['exprct_joints'][12]} {data['exprct_joints'][13]} {data['exprct_joints'][14]} {data['exprct_joints'][15]} "\
                        f"{data['exprct_joints'][16]} {data['exprct_joints'][17]} {data['exprct_joints'][18]} {data['exprct_joints'][19]} "\
                        f"{data['label']} {data['type']}\n"
                # 写入到文件中
                file.write(line)
    except Exception:
        raise

def find_grasp_lables_file(root, names, object_sets,urdf_scales):
    grasp_lables_path = os.path.join(root, 'grasp_labels')
    filenames = []
    if isinstance(names, list):
        for i in range(len(names)):
            filename = os.path.join(grasp_lables_path,object_sets[i])
            filename = os.path.join(filename,f'{names[i].stem}/{names[i].stem}_{urdf_scales[i]}.txt')
            filenames.append(filename)
        return filenames
    else:
        filename = os.path.join(grasp_lables_path,object_sets)
        filename = os.path.join(filename,f'{names.stem}/{names.stem}_{urdf_scales}.txt')
        return filename
    
    
def read_grasp_labels_stage1(filename,only_read_success=False):
    if not os.path.isfile(filename):
        if only_read_success:
            return 0
        raise FileNotFoundError(filename)
    try:
        with open(filename, 'r') as file:
            # 读取文件的每一行
            lines = file.readlines()
            # 解析第一行
            first_line = lines[0]
            # 解析第一行，假设格式为 "object_set=value name=value scale=value num_success=value"
            parts = first_line.split()
            object_set = None
            name = None
            scale = None
            num_success = None
            for part in parts:
                key, value = part.split('=')
                if key == 'object_set':
                    object_set = value
                elif key == 'name':
                    name = value
                elif key == 'scale':
                    scale = float(value)
                elif key == 'num_success':
                    num_success = int(value)
            
            if only_read_success:
                return num_success
            
            # 创建一个空列表用于存储解析的数据
            grasp_labels = []
            for line in lines[2:]:
                line = line.split()
                t7d = [float(x) for x in line[:7]]
                pre_joints = [float(x) for x in line[7:27]]
                exprct_joints = [float(x) for x in line[27:47]]
                label = int(line[47])
                type = line[48]
                data = {
                    't7d': t7d,
                    'pre_joints': pre_joints,
                    'exprct_joints': exprct_joints,
                    'label': label,
                    'type': type,
                }
                grasp_labels.append(data)
                
            return object_set, name, scale, num_success, grasp_labels
    except Exception:
        raise

def save_scene(scene_id, ids, names, object_sets, urdf_scales, poses, sim):
    try:       
        scene_path = os.path.join(DATA_ROOT,f"scenes/scene_{str(scene_id).zfill(4)}")
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
        
        obj_info = os.path.join(scene_path,'obj_info.txt')
        
        with open(obj_info , 'w') as file:

            line1 = f"ids names object_sets urdf_scales poses\n"
            file.write(line1)
            # 平面
            plane_pose = Transform(Rotation.identity(), [0, 0, 0.05]).to_list()
            line2 = f"0 data/urdfs/setup/plane.urdf None 1.0 {plane_pose[0]} {plane_pose[1]} {plane_pose[2]} {plane_pose[3]} {plane_pose[4]} {plane_pose[5]} {plane_pose[6]}\n"
            file.write(line2)
            for i in range(len(ids)):
                pose = poses[i].to_list()
                object_path = Path('data/urdfs') / object_sets[i] / Path(names[i]).name
                line =  f"{ids[i]} {object_path} {object_sets[i]} {urdf_scales[i]} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}\n"
                # 写入到文件中
                file.write(line)
                
        camero_info = os.path.join(scene_path,'cam_info.txt')
        with open(camero_info , 'w') as file:
            near = sim.camera.near
            far = sim.camera.far
            intrinsic = sim.camera.intrinsic.to_list()
            line1 = f"width height fx fy cx cy near far\n"
            file.write(line1)
            line2 = f"{intrinsic[0]} {intrinsic[1]} {intrinsic[2]} {intrinsic[3]} {intrinsic[4]} {intrinsic[5]} {near} {far}\n"
            file.write(line2)

        cam_extrinsic_file = os.path.join(scene_path,'cam_extrinsic.txt')
        
        with open(cam_extrinsic_file , 'w') as file:

            line1 = f"frame_id cam_extrinsic (qx  qy  qz  qw  x  y  z)\n"
            file.write(line1)
            
    except Exception:
        raise

def save_image_pointcloud(scene_id, frame_id, rgb, depth, seg, cam_extrinsic, sim):
    try:       
        scene_path = os.path.join(DATA_ROOT,f"scenes/scene_{str(scene_id).zfill(4)}")
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
        
        # 保存相机外参
        cam_extrinsic_file = os.path.join(scene_path,'cam_extrinsic.txt')
        with open(cam_extrinsic_file , 'a') as file:
            pose = cam_extrinsic.to_list()
            line = f"{str(frame_id).zfill(4)} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}\n"
            file.write(line)
        
        # 保存rgb图像
        rgb_path = os.path.join(DATA_ROOT,f"scenes/scene_{str(scene_id).zfill(4)}/rgb")
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)
        rgbImg = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)      
        cv2.imwrite(rgb_path+f'/{str(frame_id).zfill(4)}.png',rgbImg)
        
        # 保存深度图像
        depth_path = os.path.join(DATA_ROOT,f"scenes/scene_{str(scene_id).zfill(4)}/depth")
        if not os.path.exists(depth_path):
            os.makedirs(depth_path)
        
        depImg = depth*1000
        depImg = depImg.astype(np.uint16)

        cv2.imwrite(depth_path+f'/{str(frame_id).zfill(4)}.png',depImg)

        # 保存seg图像
        seg_path = os.path.join(DATA_ROOT,f"scenes/scene_{str(scene_id).zfill(4)}/seg")
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)     
        cv2.imwrite(seg_path+f'/{str(frame_id).zfill(4)}.png',seg)
        
    except Exception:
        raise
        
def save_stage2_grasp_lables(scene_id, grasp_labels, num_success_scene):
    try:
        # 打开文件，如果不存在则创建
        scene_path = os.path.join(DATA_ROOT,f"scenes/scene_{str(scene_id).zfill(4)}")
        # grasp_labels_path = os.path.join(scene_path,f"/grasp_labels")
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)

        filename = os.path.join(scene_path, f"grasp_lables.txt")
        
        with open(filename , 'w') as file:
            # 遍历每个抓取标签
            line1 = f"{num_success_scene}\n"
            file.write(line1)
            line2 = f"qx  qy  qz  qw  x  y  z  pre_index0  pre_index1  pre_index2  pre_index3  "\
                    f"pre_mid0  pre_mid1  pre_mid2  pre_mid3  pre_ring0  pre_ring1  pre_ring2  pre_ring3  "\
                    f"pre_little0  pre_little1  pre_little2  pre_little3  pre_thumb0  pre_thumb1  pre_thumb2  pre_thumb3  "\
                    f"index0  index1  index2  index3  mid0  mid1  mid2  mid3  "\
                    f"ring0  ring1  ring2  ring3  little0  little1  little2  little3  "\
                    f"thumb0  thumb1  thumb2  thumb3  label  type target_id (0_cebai 1_zhigeng 2_zhijian 3_zhijian)\n"
            file.write(line2)
            for data in grasp_labels:
                # 将数据转换为一行字符串
                line =  f"{data['t7d'][0]} {data['t7d'][1]} {data['t7d'][2]} {data['t7d'][3]} {data['t7d'][4]} {data['t7d'][5]} {data['t7d'][6]} "\
                        f"{data['pre_joints'][0]} {data['pre_joints'][1]} {data['pre_joints'][2]} {data['pre_joints'][3]} "\
                        f"{data['pre_joints'][4]} {data['pre_joints'][5]} {data['pre_joints'][6]} {data['pre_joints'][7]} "\
                        f"{data['pre_joints'][8]} {data['pre_joints'][9]} {data['pre_joints'][10]} {data['pre_joints'][11]} "\
                        f"{data['pre_joints'][12]} {data['pre_joints'][13]} {data['pre_joints'][14]} {data['pre_joints'][15]} "\
                        f"{data['pre_joints'][16]} {data['pre_joints'][17]} {data['pre_joints'][18]} {data['pre_joints'][19]} "\
                        f"{data['exprct_joints'][0]} {data['exprct_joints'][1]} {data['exprct_joints'][2]} {data['exprct_joints'][3]} "\
                        f"{data['exprct_joints'][4]} {data['exprct_joints'][5]} {data['exprct_joints'][6]} {data['exprct_joints'][7]} "\
                        f"{data['exprct_joints'][8]} {data['exprct_joints'][9]} {data['exprct_joints'][10]} {data['exprct_joints'][11]} "\
                        f"{data['exprct_joints'][12]} {data['exprct_joints'][13]} {data['exprct_joints'][14]} {data['exprct_joints'][15]} "\
                        f"{data['exprct_joints'][16]} {data['exprct_joints'][17]} {data['exprct_joints'][18]} {data['exprct_joints'][19]} "\
                        f"{data['label']} {data['type']} {data['target_id']}\n"
                # 写入到文件中
                file.write(line)
    except Exception:
        raise

def save_visible_mask(scene_id, frame_id, visible_mask):
    try:       
        scene_path = os.path.join(DATA_ROOT,f"scenes/scene_{str(scene_id).zfill(4)}")
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
    
        filename = os.path.join(scene_path, f"visible_mask.txt")
        with open(filename , 'a') as file:
            line = f"{str(frame_id).zfill(4)} {visible_mask}\n"
            file.write(line)
    
    except Exception:
        raise
