import os
import pdb
import sys
import cv2
import numpy as np
import torch
import open3d as o3d
import copy
import open3d.visualization.rendering as rendering
from vision_3d.camera_info import INTRINSICS_REALSENSE_1280, INTRINSICS_CLIP_VIEW
from PIL import Image
from tqdm import tqdm
from vis_utils import visimg
from pytorch3d.transforms import euler_angles_to_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_vis_pcds(rgbs, depths, cam_poses, intrinsics, masks, num_objs, scene_bounds,
                 save_dir=None, vis=False, use_cache=True, pcds_type=1, single_view_idx=0):
    if use_cache:
        print('Using cached visual point cloud models')
        obj_pcds = []
        for obj_id in range(num_objs):
            obj_pcd = o3d.io.read_point_cloud(f'{save_dir}/obj_vis_{obj_id}.pcd')
            obj_pcds.append(obj_pcd)
        if vis:
            for obj_id in range(num_objs):
                obj_pcd = obj_pcds[obj_id]
            o3d.visualization.draw_geometries(obj_pcds)
        return obj_pcds

    print('Creating visual point cloud models...')
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if pcds_type == 0:
        # Single view pcd
        view_num = 1
    else:
        # Multi view pcd
        view_num = len(depths)

    crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=scene_bounds[0], max_bound=scene_bounds[1])
    frame_voxel_size = 0.002
    obj_voxel_size = 0.002
    outlier_neighbours = 30
    outlier_std_ratio = 1.05
    obj_pcds = []
    for obj_id in range(num_objs):
        obj_pcd = o3d.geometry.PointCloud()
        view_range = range(view_num) if pcds_type == 1 else [single_view_idx]
        for frame_id in view_range:
            depth = depths[frame_id].clone().cpu().numpy()
            rgb = rgbs[frame_id].clone().cpu().numpy()
            cam_pose = cam_poses[frame_id].cpu().numpy()
            mask = masks[frame_id].clone()
            mask = mask == obj_id

            # Erode mask to counter outliers due to imperfections in masks / depth measurements at obj edges.
            # Note that this does not really do anything for the task bground object because it is all one task bground mask anyway.
            mask = mask.cpu().numpy().astype(np.uint8)
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1).astype(np.bool)

            depth[~mask] = 0
            rgb[~mask] = 0
            height = depth.shape[0]
            width = depth.shape[1]
            depth = o3d.geometry.Image((depth * 1000).astype(np.uint16))
            rgb = o3d.geometry.Image(rgb.astype(np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1000, depth_trunc=1000, convert_rgb_to_intensity=False)
            T_cw = np.linalg.inv(cam_pose)
            o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsics)
            frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics, T_cw)
            frame_pcd = frame_pcd.crop(crop_bbox)
            if pcds_type == 1:
                frame_pcd = frame_pcd.voxel_down_sample(frame_voxel_size)
            obj_pcd += frame_pcd

        # _, inlier_idxs = obj_pcd.remove_statistical_outlier(nb_neighbors=outlier_neighbours, std_ratio=outlier_std_ratio)
        # obj_pcd = obj_pcd.select_by_index(inlier_idxs)
        # obj_pcd = obj_pcd.voxel_down_sample(obj_voxel_size)
        obj_pcds.append(obj_pcd)

        if save_dir is not None:
            pcd_out_path = os.path.join(save_dir, f'obj_vis_{obj_id}.pcd')
            o3d.io.write_point_cloud(pcd_out_path, obj_pcd)

    if vis:
        for obj_id in range(num_objs):
            obj_pcd = obj_pcds[obj_id]
        o3d.visualization.draw_geometries(obj_pcds)

    print('Visual point cloud models created.')
    return obj_pcds


class PointCloudRenderer():
    def __init__(self):
        width, height = 336, 336
        self.renderer = rendering.OffscreenRenderer(width, height)
        # K = INTRINSICS_REALSENSE_1280
        K = INTRINSICS_CLIP_VIEW
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        self.mat = rendering.MaterialRecord()
        self.mat.point_size = 3.0

    # Returns list of images, one for each pose of the movable object.
    # Does not expect scene_model.bground_obj to be in relevant_objs but will render anyway.
    # OPT: we can probably reduce how often we move tensors to/from GPU.
    def render(self, render_pose, pose_batch, task_model, hide_movable=False):
        cam_pose = render_pose
        cam_pose_inv = np.linalg.inv(cam_pose)
        extrinsics = cam_pose_inv
        self.renderer.setup_camera(self.intrinsics, extrinsics)

        # Only need to add bground object once.
        self.renderer.scene.add_geometry(task_model.task_bground_obj.name, task_model.task_bground_obj.vis_model, self.mat)

        colours = []
        # depths = []

        if not hide_movable:
            if pose_batch.shape[0] > 1:
                print('Rendering scene point cloud for each pose of movable object...')
            for pose_idx in tqdm(range(pose_batch.shape[0]), disable=(pose_batch.shape[0] == 1)):
                # OPT: batch this.
                # Update pose of movable object.
                old_pose = task_model.movable_obj.pose.cpu()
                sample_pose = pose_batch[pose_idx].reshape(4, 4).cpu()
                pose_transform = sample_pose @ old_pose.inverse()
                old_movable_pcd = task_model.movable_obj.vis_model
                movable_pcd = copy.deepcopy(old_movable_pcd)
                movable_pcd.transform(pose_transform.cpu().numpy())
                self.renderer.scene.add_geometry(task_model.movable_obj.name, movable_pcd, self.mat)

                colour = np.asarray(self.renderer.render_to_image())
                # depth = np.asarray(renderer.render_to_depth_image())

                self.renderer.scene.remove_geometry(task_model.movable_obj.name)

                # visimg(colour)
                # visimg(depth)

                # Use black background for fair comparison with other methods.
                # Select all white pixels (those where all RGB channels over 220) and set to black.
                colour[np.all(colour > 220, axis=-1)] = 0

                colours.append(colour)
                # depths.append(depth)
        else:
            raise NotImplementedError

        self.renderer.scene.remove_geometry(task_model.task_bground_obj.name)
        return colours