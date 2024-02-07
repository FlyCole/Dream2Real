#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import pdb

import commentjson as json

import numpy as np

import shutil
import time
import sys
sys.path.append('./reconstruction/instant-ngp/build')
sys.path.append('./reconstruction/instant-ngp/scripts')
from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa
import cv2

from vision_3d.camera_info import INTRINSICS_CLIP_VIEW
import utils.accio2ngp as accio2ngp


class obj_nerf:
	"""
	Object Nerf class for unit test
	"""
	def __init__(self, nerf_file):
		self.testbed = ngp.Testbed()
		self.testbed.root_dir = ROOT_DIR
		self.testbed.load_file(nerf_file)

		self.testbed.background_color = [1.0, 1.0, 1.0, 0.0]
		self.testbed.snap_to_pixel_centers = True
		self.spp = 1

		self.testbed.nerf.render_min_transmittance = 1e-4

		self.testbed.shall_train = False


class renderer:
	def __init__(self, data_dir, task_model=None):
		self.root = data_dir

		if task_model is not None:
			self.bg_obj = task_model.task_bground_obj
			self.fg_obj = task_model.movable_obj
		else:
			# for unit test
			self.bg_obj = obj_nerf(os.path.join(self.root, "bg_base.ingp"))
			self.fg_obj = obj_nerf(os.path.join(self.root, "fg_base.ingp"))
			self.bg_render_file = os.path.join(self.root, "bg_render_transforms.json")
			self.fg_render_file = os.path.join(self.root, "fg_render_transforms.json")
			self.convert_poses()

		# Output setup
		self.out_render_path = os.path.join(self.root, "cb_render")
		os.makedirs(self.out_render_path, exist_ok=True)

	def render(self, valid_poses, render_poses, render_cam_pose_idx, depths_gt=None, movable_masks=None, save=True):
		"""
		Render function in whole pipeline for combined rendering using instant-ngp.
		:param valid_poses: valid poses list [K, 4, 4] (np.array)
		:param render_poses: camera render pose in ngp convention [L, 4, 4] (np.array)
		:param depths_gt: depth ground truth map for background rendering [:, H, W] (torch.Tensor)
		:param movable_masks: masks of movable objects [N, H, W] (list(torch.Tensor))
		:return: rendered image [K x L, H, W, 4] (list(torch.Tensor))
		"""
		T_WO_1 = accio2ngp.converter(np.expand_dims(self.fg_obj.pose.cpu().numpy(), axis=0))
		render_imgs = []

		# Set camera
		resolution = [336, 336]

		if save:
			# Delete old renders first.
			if os.path.exists(self.out_render_path):
				shutil.rmtree(self.out_render_path)
			os.makedirs(self.out_render_path)

		# Background rendering
		for render_idx in range(len(render_cam_pose_idx)):
			cam_matrix = render_poses[render_idx]

			self.bg_obj.vis_model.set_camera_to_training_view(render_cam_pose_idx[render_idx])
			self.bg_obj.vis_model.background_color = [0.0, 0.0, 0.0, 1.0]

			self.bg_obj.vis_model.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1, :])
			self.bg_obj.vis_model.render_ground_truth = False

			self.bg_obj.vis_model.render_mode = ngp.Shade
			bg_image = self.bg_obj.vis_model.render(resolution[0], resolution[1], 1, True)

			if depths_gt is not None:
				bg_depth = self.rectify_depth(depths_gt[render_idx], resolution)
				bg_mask = self.rectify_mask(movable_masks[render_idx], resolution)
				bg_depth[bg_mask == 0, 0] = 100
			else:
				self.bg_obj.vis_model.render_mode = ngp.Depth
				bg_depth = self.bg_obj.vis_model.render(resolution[0], resolution[1], 1, True)

			# Foreground rendering
			self.fg_obj.vis_model.set_camera_to_training_view(render_cam_pose_idx[render_idx])
			with tqdm(range(len(valid_poses)), unit="images", desc=f"Rendering test frame") as t:
				for i in t:
					# Foreground rendering
					# Set camera
					resolution = [336, 336]
					cam_matrix = convert_virtual_pose(T_WO_1, valid_poses[i], render_poses[render_idx])
					self.fg_obj.vis_model.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1, :])

					self.fg_obj.vis_model.render_ground_truth = False
					self.fg_obj.vis_model.render_mode = ngp.Shade
					fg_image = self.fg_obj.vis_model.render(resolution[0], resolution[1], 1, True)

					self.fg_obj.vis_model.render_mode = ngp.Depth
					fg_depth = self.fg_obj.vis_model.render(resolution[0], resolution[1], 1, True)

					# Combined rendering
					cb_image = bg_image.copy()
					fg_depth[fg_depth[..., 0] < 0.05, 0] = 100
					bg_depth[bg_depth[..., 0] < 0.05, 0] = 100
					near_depth_map = fg_depth[..., 0] < bg_depth[..., 0]

					final_map = near_depth_map
					# cv2.imshow("near", near_depth_map.astype(np.uint8) * 255)
					# cv2.imshow("alpha", alpha_check_map.astype(np.uint8) * 255)
					# cv2.imshow("final", final_map.astype(np.uint8) * 255)
					# cv2.waitKey(1)

					cb_image[final_map, :] = fg_image[final_map, :]

					# change the format of cb_image to align with the following code
					img = np.copy(cb_image)
					img[..., 0:3] = np.divide(img[..., 0:3], img[..., 3:4], out=np.zeros_like(img[..., 0:3]),
											where=img[..., 3:4] != 0)
					img[..., 0:3] = linear_to_srgb(img[..., 0:3])
					img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
					# Select pixels where alpha value less than threshold and set them to black.
					img[img[..., 3] < 130, :] = 0
					img_out = img[..., :3]
					render_imgs.append(img_out)

		if save and render_idx == 0:
			for i in tqdm(range(len(render_imgs)), unit="images", desc=f"Writing renders to disk"):
				cv2.imwrite(os.path.join(self.out_render_path, f"cb_rgb_{i:04d}.png"), cv2.cvtColor(render_imgs[i], cv2.COLOR_RGB2BGR))
				# if video_save:
				# 	write_image(os.path.join(video_folder, f"fg_rgb_{i:04d}.png"), render_fg_imgs[i])

		return render_imgs


	def rectify_depth(self, depth_ori, resolution):
		"""
		Rectify depth map to make sure the depth is rendered by CLIP view camera
		:param depth_ori: original depths map list [H_ori, W_ori] (torch.Tensor)
		:param resolution: resolution of the rectified depth map [H_clip, W_clip] (list)
		:return: depth_clip: rectified depth map [H_clip, W_clip, 4] (np.array)
		"""
		img = depth_ori.cpu().numpy()

		# center crop
		h, w = img.shape[:2]
		if h > w:
			img = img[(h - w) // 2:(h - w) // 2 + w, :]
		else:
			img = img[:, (w - h) // 2:(w - h) // 2 + h]

		# transform from numpy to cv2 image
		img_cv = img.astype(np.float32)
		depth_clip = cv2.resize(img_cv, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
		depth_clip = np.repeat(np.expand_dims(depth_clip, axis=2), 4, axis=2)

		return depth_clip

	def rectify_mask(self, mask_ori, resolution):
		"""
		Rectify mask map to make sure the depth is rendered by CLIP view camera
		:param mask_ori: original mask map list [H_ori, W_ori] (torch.Tensor)
		:param resolution: resolution of the rectified mask map [H_clip, W_clip] (list)
		:return: mask_clip: rectified mask map [H_clip, W_clip] (np.array)
		"""
		img = mask_ori.cpu().numpy()

		# center crop
		h, w = img.shape[:2]
		if h > w:
			img = img[(h - w) // 2:(h - w) // 2 + w, :]
		else:
			img = img[:, (w - h) // 2:(w - h) // 2 + h]

		# transform from numpy to cv2 image
		img_cv = img.astype(np.uint8)
		mask_clip = cv2.resize(img_cv, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)

		return mask_clip

	def convert_poses(self):
		# Background render transforms
		transforms_file = os.path.join(self.root, "bg_transforms.json")
		output_file = os.path.join(self.root, "bg_render_transforms.json")
		with open(transforms_file) as json_file:
			out = json.load(json_file)
		out["fl_x"] = INTRINSICS_CLIP_VIEW[0, 0]
		out["fl_y"] = INTRINSICS_CLIP_VIEW[1, 1]
		out["cx"] = INTRINSICS_CLIP_VIEW[0, 2]
		out["cy"] = INTRINSICS_CLIP_VIEW[1, 2]
		out["w"] = 336
		out["h"] = 336
		out["frames"] = [out["frames"][0]]

		with open(output_file, "w") as outfile:
			json.dump(out, outfile, indent=2)

		# Foreground render transforms
		transforms_file = os.path.join(self.root, "fg_transforms.json")
		output_file = os.path.join(self.root, "fg_render_transforms.json")
		with open(transforms_file) as json_file:
			out = json.load(json_file)
		out["fl_x"] = INTRINSICS_CLIP_VIEW[0, 0]
		out["fl_y"] = INTRINSICS_CLIP_VIEW[1, 1]
		out["cx"] = INTRINSICS_CLIP_VIEW[0, 2]
		out["cy"] = INTRINSICS_CLIP_VIEW[1, 2]
		out["w"] = 336
		out["h"] = 336
		matrix_0 = np.array(out["frames"][0]["transform_matrix"])
		for i in range(len(out["frames"])):
			matrix = matrix_0.copy()
			matrix[2, 3] -= 0.02 * i
			matrix[1, 3] -= 0.02 * i
			out["frames"][i]["transform_matrix"] = matrix.tolist()

		with open(output_file, "w") as outfile:
			json.dump(out, outfile, indent=2)


def convert_virtual_pose(T_WO_1, T_WO_2, T_WC_1):
	"""
	Convert to virtual camera pose in order to make T_C1_O2 = T_C2_O1,
	meaning that the target object pose in camera frame is the same
	as the current object pose in the virtual camera frame.
	:param T_WO_1: Object pose in world frame
	:param T_WO_2: Target object pose in world frame
	:param T_WC_1: Camera pose in world frame
	:return: T_WC_2: Virtual camera pose in world frame
	"""
	T_O2_O1 = np.linalg.inv(T_WO_2) @ T_WO_1
	T_O1_C1 = np.linalg.inv(T_WO_1) @ T_WC_1
	T_WC_2 = T_WO_1 @ T_O2_O1 @ T_O1_C1
	return T_WC_2


if __name__ == "__main__":
	renderer = renderer()







