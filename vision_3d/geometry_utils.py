import copy
import cv2
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from vis_utils import visimg
from vision_3d.camera_info import INTRINSICS_REALSENSE_1280
from vis_utils import pastel_colors
import matplotlib.pyplot as plt
import torchvision

import pdb

# NOTE: depth_img should be in mm.
# cam_pose is 4x4 transformation matrix. Check for inv before passing in.
# Returns pcd in metres.
def obj_pcd_from_depth_and_mask(depth_img, cam_pose, intrinsics):
    depth_img = depth_img.copy()
    depth_img[depth_img == 0] = 1 # Set invalid depth to be very close, 1 mm away.
    depth_trunc_metres = 1000.0
    assert depth_img.dtype == np.uint16
    width = depth_img.shape[1]
    height = depth_img.shape[0]
    depth_img = o3d.geometry.Image(depth_img)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsics)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img, intrinsics, cam_pose, depth_scale=1000, depth_trunc=depth_trunc_metres, project_valid_depth_only=True)
    assert len(pcd.points) == width * height # Important, otherwise point selection may not work afterwards.
    return pcd

def downsample_pcd(pcd, voxel_size):
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return down_pcd

# OPT: can do voxel downsampling first.
# Computes normals facing towards camera.
# NOTE: in-place.
def estimate_normals(pcd, cam_pose):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=300))
    pcd.orient_normals_towards_camera_location(cam_pose[:3, 3])
    return pcd

# NOTE: assumes that pcd has not been downsampled.
def project_pix_to_pcd(row, col, img_height, img_width, pcd, visualise_grasp=False):
    index = row * img_width + col
    pos = pcd.points[index].copy()
    normal = pcd.normals[index].copy()

    # Recolor that point to red.
    if visualise_grasp:
        colors = np.asarray(pcd.colors)
        colors[index] = [1, 0, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pos, normal

def get_grasp_pose(pos, normal):
    pseudo_up = np.array([0, 1, 0])
    if np.abs(np.dot(normal, pseudo_up)) > 0.99:
        # This means that the normal is almost parallel to the pseudo_up vector.
        # Choose another pseudo_up vector.
        pseudo_up = np.array([0, 0, 1])
    z_axis = -1 * normal # Based on z-axis direction in tipLink frame (we want gripper to be pointing towards object).
    x_axis = pseudo_up
    y_axis = np.cross(z_axis, x_axis)
    # Now get proper x-axis which is perpendicular to z-axis and y-axis.
    x_axis = -1 * np.cross(z_axis, y_axis)

    # Create rotation matrix.
    grasp_orientation = np.eye(4)
    grasp_orientation[:3, 0] = x_axis / np.linalg.norm(x_axis)
    grasp_orientation[:3, 1] = y_axis / np.linalg.norm(y_axis)
    grasp_orientation[:3, 2] = z_axis / np.linalg.norm(z_axis)

    # Populate the transformation matrix.
    grasp_pose = np.eye(4)
    grasp_pose[:3, :3] = grasp_orientation[:3, :3]
    grasp_pose[:3, 3] = pos
    return grasp_pose

# Rotation by angle_deg is applied clockwise about the normal vector.
def rotate_grasp_about_normal(grasp_pose, normal_axis, angle_deg):
    rotation_matrix = cv2.Rodrigues(normal_axis * np.deg2rad(angle_deg))[0]
    grasp_ori = np.matmul(rotation_matrix, grasp_pose[:3, :3])

    new_grasp_pose = np.eye(4)
    new_grasp_pose[:3, :3] = grasp_ori
    new_grasp_pose[:3, 3] = grasp_pose[:3, 3]
    return new_grasp_pose

def normalise_angle(angle_deg):
    angle_deg = np.rad2deg(np.arctan2(np.sin(np.deg2rad(angle_deg)), np.cos(np.deg2rad(angle_deg))))
    return angle_deg

# NOTE: still sometimes returns nan values from cv2.inpaint, so we set those to zero.
def patch_up_depth(depth_img):
    depth_img = depth_img.copy()
    depth_mask = (np.logical_or(np.isnan(depth_img), (depth_img == 0))).astype(np.uint8)
    patched_depth = cv2.inpaint(depth_img, depth_mask, 3, cv2.INPAINT_NS)
    depth_img[depth_mask == 1] = patched_depth[depth_mask == 1]
    depth_img[np.isnan(depth_img)] = 0
    return depth_img

# Assume depth_img is already patched up.
# cam_pose is transformation matrix.
def get_grasp_pose_from_pix(row, col, depth_img, intrinsics, cam_pose, visualise_grasp=False):
    if depth_img[row, col] == 0:
        raise ValueError("Attempted grasp at pixel with zero depth value.")
    inv_cam_pose = np.linalg.inv(cam_pose)
    pcd = obj_pcd_from_depth_and_mask(depth_img, inv_cam_pose, intrinsics)
    if visualise_grasp:
        grey_color = [0.5, 0.5, 0.5]
        pcd.colors = o3d.utility.Vector3dVector(np.tile(grey_color, (len(pcd.points), 1)))
    pcd = estimate_normals(pcd, inv_cam_pose)
    pos, normal = project_pix_to_pcd(row, col, depth_img.shape[0], depth_img.shape[1], pcd, visualise_grasp=visualise_grasp)
    normal = np.array([0, 0, 1])
    grasp_pose = get_grasp_pose(pos, normal)
    if visualise_grasp:
        frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame_vis.transform(grasp_pose)
        o3d.visualization.draw_geometries([pcd, frame_vis], point_show_normal=False)
    return grasp_pose, normal

# Takes in pixel coordinates (row and col, i.e. i then j), height of image, depth at that point, cam_pose (extrinsics matrix), intrinsics.
# Outputs 3D position in world coordinates.
def pix_to_world(row, col, img_height, depth, cam_pose, intrinsics):
    assert depth is not None
    u, v = (img_height - row), col
    intrinsics_inv = np.linalg.inv(intrinsics)
    cam_frame_coords = intrinsics_inv @ (depth * np.array([u, v, 1]))
    cam_frame_coords = np.append(cam_frame_coords, 1)
    world_frame_coords = cam_pose @ cam_frame_coords
    world_frame_coords = (world_frame_coords / world_frame_coords[3])[:3]
    return world_frame_coords

def vis_cost_volume(pose_scores, sample_res, pose_batch, bground_geoms, use_phys_mods=True, colourmap=True, fixed_ori=None, exp=True):
    # Normalise scores for visualisation.
    pose_scores = pose_scores.clone().double()
    nonzero_idxs = torch.nonzero(pose_scores, as_tuple=True)
    zero_idxs = torch.nonzero(pose_scores == 0, as_tuple=True)
    if exp:
        # Raise to higher power, to better visualise differences between high-scoring ones.
        pose_scores[nonzero_idxs] = 10 ** (pose_scores[nonzero_idxs] * 10)
    min_nonzero = torch.min(pose_scores[nonzero_idxs])
    pose_scores[nonzero_idxs] = (pose_scores[nonzero_idxs] - min_nonzero) / (torch.max(pose_scores[nonzero_idxs]) - min_nonzero)

    # Get score for each pos by maxing over all orientations.
    pos_scores = pose_scores.view(sample_res[0] * sample_res[1] * sample_res[2], sample_res[3], sample_res[4], sample_res[5])
    pos_scores = pos_scores.view(sample_res[0] * sample_res[1] * sample_res[2], sample_res[3] * sample_res[4] * sample_res[5])
    pos_scores, max_score_idx = torch.max(pos_scores, dim=1)
    pos_batch = pose_batch.view(sample_res[0] * sample_res[1] * sample_res[2], sample_res[3], sample_res[4], sample_res[5], 4, 4)
    pos_batch = pos_batch.view(sample_res[0] * sample_res[1] * sample_res[2], sample_res[3] * sample_res[4] * sample_res[5], 4, 4)
    pos_batch = pos_batch[torch.arange(pos_batch.shape[0]), max_score_idx]
    pos_batch = pos_batch.squeeze()
    pos_batch = pos_batch[:, :3, 3].squeeze()

    min_x, max_x = torch.min(pos_batch[:, 0]).cpu().numpy(), torch.max(pos_batch[:, 0]).cpu().numpy()
    min_y, max_y = torch.min(pos_batch[:, 1]).cpu().numpy(), torch.max(pos_batch[:, 1]).cpu().numpy()
    min_z, max_z = torch.min(pos_batch[:, 2]).cpu().numpy(), torch.max(pos_batch[:, 2]).cpu().numpy()
    pos_batch = pos_batch.cpu().numpy()
    inter_x_dist = (max_x - min_x) / (sample_res[0] - 1) # This is also the voxel size.
    inter_y_dist = (max_y - min_y) / (sample_res[1] - 1)
    inter_z_dist = (max_z - min_z) / (sample_res[2] - 1)
    # x_width, y_width, z_width = inter_x_dist, inter_y_dist, inter_z_dist
    min_inter_dist = min(inter_x_dist, inter_y_dist, inter_z_dist)
    x_width, y_width, z_width = min_inter_dist, min_inter_dist, min_inter_dist

    # bground_mat = o3d.visualization.rendering.MaterialRecord()
    # bground_mat.shader = "defaultUnlit"
    bground_mat = None
    if use_phys_mods:
        meshes = [o3d.io.read_triangle_mesh(mesh_path) for mesh_path in bground_geoms]
        if bground_mat is None:
            geometries = [{'name': f'bground_mesh_{i:03}', 'geometry': mesh} for i, mesh in enumerate(meshes)]
        else:
            geometries = [{'name': f'bground_mesh_{i:03}', 'geometry': mesh, 'material': bground_mat} for i, mesh in enumerate(meshes)]
        for i, mesh in enumerate(meshes):
            mesh.compute_vertex_normals()
            # col = pastel_colors[i % pastel_colors.shape[0]] / 255.0
            col = np.array([100, 100, 100]) / 255.0
            mesh.paint_uniform_color(col)
    else:
        geometries = [{'name': 'bground_vis', 'geometry': bground_vis}]
    pos_scores_flattened = pos_scores.view(-1).cpu().numpy()
    if colourmap:
        cmap = plt.get_cmap('viridis')
        # use more vibrant colours.
    boxes = []
    for i in range(pos_batch.shape[0]):
        score = pos_scores_flattened[i]
        if score < 0.001:
            continue
        pos = pos_batch[i]
        box = o3d.geometry.TriangleMesh.create_box(width=x_width, height=y_width, depth=z_width)
        box.translate(pos + np.array([-1 * x_width / 2, y_width / 2, -1 * z_width / 2]))
        color = cmap(score)
        box.paint_uniform_color([color[0], color[1], color[2]])
        boxes.append(box)
    all_boxes = boxes[0]
    for box in boxes[1:]:
        all_boxes += box
    box_mat = o3d.visualization.rendering.MaterialRecord()
    box_mat.shader = "defaultUnlit"
    all_boxes_geom = {'name': 'all_boxes', 'geometry': all_boxes, 'material': box_mat}
    geometries.append(all_boxes_geom)
    o3d.visualization.draw(geometries, show_skybox=False)

def vis_multiverse(pose_scores, sample_res, pose_batch, bground_geoms, movable_geom, movable_init_pose, use_phys_mods=True, colourmap=True):
    # Normalise scores for visualisation.
    pose_scores = pose_scores.clone()
    nonzero_idxs = torch.nonzero(pose_scores, as_tuple=True)
    min_nonzero = torch.min(pose_scores[nonzero_idxs])
    pose_scores[nonzero_idxs] = (pose_scores[nonzero_idxs] - min_nonzero) / (torch.max(pose_scores[nonzero_idxs]) - min_nonzero)

    if use_phys_mods:
        meshes = [o3d.io.read_triangle_mesh(mesh_path) for mesh_path in bground_geoms]
        geometries = [{'name': f'bground_mesh_{i:03}', 'geometry': mesh} for i, mesh in enumerate(meshes)]
        for i, mesh in enumerate(meshes):
            mesh.compute_vertex_normals()
            col = pastel_colors[i % pastel_colors.shape[0]] / 255.0
            mesh.paint_uniform_color(col)
    else:
        geometries = [{'name': 'bground_vis', 'geometry': bground_vis}]
    pose_scores_flattened = pose_scores.view(-1).cpu().numpy()
    if colourmap:
        cmap = plt.get_cmap('viridis')
        movable_geom = o3d.io.read_triangle_mesh(movable_geom)
    pose_batch = pose_batch.to('cpu').view(-1, 4, 4).numpy()
    movable_init_pose = movable_init_pose.to('cpu').view(4, 4).numpy()
    init_inv = np.linalg.inv(movable_init_pose)
    movable_geom.compute_vertex_normals()
    for i in range(0, pose_batch.shape[0], 7):
        score = pose_scores_flattened[i]
        if score == 0:
            continue
        pose = pose_batch[i]
        pose_transform = pose @ init_inv
        curr_movable_geom = copy.deepcopy(movable_geom)
        curr_movable_geom.transform(pose_transform)
        material = o3d.visualization.rendering.MaterialRecord()
        color = [140 / 255, 251 / 255, 140 / 255]
        material.shader = 'defaultLitTransparency'
        material.base_color = np.array([color[0], color[1], color[2], score])
        name = f'pose({pose}), score {score:.2f}'
        geometry = {'name': name, 'geometry': curr_movable_geom, 'material': material}
        geometries.append(geometry)

    o3d.visualization.draw(geometries, show_skybox=False)

# pose_scores should have shape should have shape (sample_res.prod()).
def spatially_smooth_heatmap(pose_scores, sample_res, sigma=0.7):
    pose_scores = pose_scores.clone()
    min_nonzero = torch.min(pose_scores[pose_scores != 0]).item()
    zero_idxs = torch.nonzero(pose_scores == 0, as_tuple=True)
    pose_scores[zero_idxs] = min_nonzero
    pose_scores = pose_scores.view(sample_res[0] * sample_res[1], sample_res[2] * sample_res[3] * sample_res[4] * sample_res[5])
    pose_scores = pose_scores.swapaxes(0, 1)
    pose_scores = pose_scores.unsqueeze(1)
    pose_scores = pose_scores.view(sample_res[2] * sample_res[3] * sample_res[4] * sample_res[5], 1, sample_res[0], sample_res[1]) # Make it image-batch-shaped.
    pose_scores = torchvision.transforms.functional.pad(pose_scores, padding=1, fill=min_nonzero, padding_mode='constant')
    smoothed_scores = torchvision.transforms.functional.gaussian_blur(pose_scores, kernel_size=3, sigma=sigma)
    smoothed_scores = smoothed_scores[:, :, 1:-1, 1:-1] # Remove the padding again.
    smoothed_scores = smoothed_scores.reshape(sample_res[2] * sample_res[3] * sample_res[4] * sample_res[5], 1, sample_res[0] * sample_res[1])
    smoothed_scores = smoothed_scores.squeeze(1)
    smoothed_scores = smoothed_scores.swapaxes(0, 1)
    smoothed_scores = smoothed_scores.reshape(sample_res[0] * sample_res[1] * sample_res[2] * sample_res[3] * sample_res[4] * sample_res[5])
    smoothed_scores[zero_idxs] = 0
    return smoothed_scores.contiguous()
