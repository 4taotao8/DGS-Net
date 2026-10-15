"""Open3D visualization of predicted HITDLR3 grasp and pre-grasp hands."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


HAND_URDF = Path(__file__).resolve().parent.parent / 'assets/hand/hand_right_simplify.urdf'
FINGERS = ('index', 'mid', 'ring', 'little', 'thumb')


def transform(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation.as_matrix()
    if translation is not None:
        result[:3, 3] = translation
    return result


def origin_transform(element):
    if element is None:
        return np.eye(4)
    xyz = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    return transform(Rotation.from_euler('xyz', rpy), xyz)


class HandMeshes:
    """Read URDF visual meshes once and pose them with its joint hierarchy."""

    def __init__(self, urdf=HAND_URDF):
        self.urdf = Path(urdf)
        root = ET.parse(self.urdf).getroot()
        self.visuals = {}
        for link in root.findall('link'):
            visuals = []
            for visual in link.findall('visual'):
                mesh_element = visual.find('geometry/mesh')
                if mesh_element is None:
                    continue
                mesh_path = self.urdf.parent / mesh_element.get('filename')
                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                if mesh.is_empty():
                    raise ValueError(f'Could not load hand mesh: {mesh_path}')
                mesh.compute_vertex_normals()
                scale = np.fromstring(mesh_element.get('scale', '1 1 1'), sep=' ')
                mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
                visuals.append((mesh, origin_transform(visual.find('origin'))))
            self.visuals[link.get('name')] = visuals
        self.children = {}
        for joint in root.findall('joint'):
            parent = joint.find('parent').get('link')
            self.children.setdefault(parent, []).append({
                'name': joint.get('name'),
                'type': joint.get('type'),
                'child': joint.find('child').get('link'),
                'origin': origin_transform(joint.find('origin')),
                'axis': np.fromstring(joint.find('axis').get('xyz', '0 0 1'), sep=' '),
            })
        self.tcp_to_body = np.linalg.inv(transform(
            Rotation.from_quat([0, 0.21676, 0, 0.97622]),
            [0.0421128868793952, 0, 0.129018711905046],
        ))

    def pose(self, tcp_rotation, tcp_position, joints):
        """Return world-frame visual meshes for one 20-joint grasp."""
        values = np.clip(np.asarray(joints).reshape(5, 4),
                         [-0.35, 0, 0, 0], [0.35, 1.48, 1.4, 1.4])
        angles = {f'{finger}_joint_Link{part + 1}': values[finger_id, part]
                  for finger_id, finger in enumerate(FINGERS) for part in range(4)}
        root_pose = transform(Rotation.from_matrix(tcp_rotation), tcp_position) @ self.tcp_to_body
        output = []

        def visit(link, link_pose):
            for template, visual_pose in self.visuals.get(link, []):
                mesh = deepcopy(template)
                mesh.transform(link_pose @ visual_pose)
                output.append(mesh)
            for joint in self.children.get(link, []):
                child_pose = link_pose @ joint['origin']
                if joint['type'] in ('revolute', 'continuous'):
                    child_pose = child_pose @ transform(
                        Rotation.from_rotvec(joint['axis'] * angles[joint['name']]))
                visit(joint['child'], child_pose)

        visit('base_link', root_pose)
        return output


def sphere(position, color, radius=0.005):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.compute_vertex_normals()
    mesh.translate(position)
    mesh.paint_uniform_color(color)
    return mesh


def material(color, alpha=1.0):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultLitTransparency' if alpha < 1 else 'defaultLit'
    mat.base_color = [*color, alpha]
    return mat


def show_grasps(cloud, grasps, count=4, seed=None):
    """Show random post-NMS grasp and pre-grasp hand meshes in one Open3D view."""
    if grasps is None or not len(grasps['score']):
        print('No grasps to visualize.')
        return
    indices = np.random.default_rng(seed).choice(len(grasps['score']),
                                                  size=min(count, len(grasps['score'])), replace=False)
    hand = HandMeshes()
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    viewer = o3d.visualization.O3DVisualizer('DGS-Net predicted grasps', 1280, 720)
    viewer.set_background((1.0, 1.0, 1.0, 1.0), None)
    viewer.show_skybox(False)
    viewer.enable_raw_mode(True)
    viewer.add_geometry({'name': 'scene', 'geometry': cloud})
    final_color = [0.18, 0.48, 0.75]
    pre_color = [0.67, 0.78, 0.87]
    for rank, index in enumerate(indices):
        tcp = grasps['position'][index]
        rotation = grasps['rotation'][index]
        offset = grasps['offset'][index]
        pre_tcp = tcp - 0.1 * rotation[:, 2]
        for part, mesh in enumerate(hand.pose(rotation, tcp, grasps['final_joints'][index])):
            viewer.add_geometry({'name': f'grasp_{rank}_hand_{part}', 'geometry': mesh,
                                 'material': material(final_color)})
        for part, mesh in enumerate(hand.pose(rotation, pre_tcp, grasps['pre_joints'][index])):
            viewer.add_geometry({'name': f'grasp_{rank}_pre_{part}', 'geometry': mesh,
                                 'material': material(pre_color, 0.45)})
        viewer.add_geometry({'name': f'grasp_{rank}_tcp',
                             'geometry': sphere(tcp, [0.5, 0, 0.5])})
        viewer.add_geometry({'name': f'grasp_{rank}_observed_point',
                             'geometry': sphere(tcp + offset, [1, 0, 0])})
        print(f'visualized grasp {index}: score={grasps["score"][index]:.3f}, TCP={tcp}')
    viewer.reset_camera_to_default()
    app.add_window(viewer)
    app.run()
