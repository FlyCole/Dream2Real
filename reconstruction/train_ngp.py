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
import pathlib
curr_dir_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(os.path.join(curr_dir_path, 'instant-ngp/build'))
sys.path.append(os.path.join(curr_dir_path, 'instant-ngp/scripts'))
from common import *
from scenes import *
from cfg import Config

from tqdm import tqdm

import pyngp as ngp # noqa


def get_scene(scene):
	for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
		if scene in scenes:
			return scenes[scene]
	return None


def build_vis_model(cfg, dynamic_time_extension=True, render_distract=False):
	testbed = ngp.Testbed()
	testbed.root_dir = ROOT_DIR

	file = cfg.files
	# If the transforms.json file doesn't exist, try to convert from optimised pose to instant-ngp format
	if cfg.optimize_extrinsics:
		print("Trying to convert from optimised pose to instant-ngp format...")
		from utils import accio2ngp
		accio2ngp.raw_poses_convert(cfg, file)

	scene_info = get_scene(file)
	if scene_info:
		file = os.path.join(scene_info["data_dir"], scene_info["dataset"])
	testbed.load_file(file)

	# Pick a sensible GUI resolution depending on arguments.
	if cfg.gui:
		testbed.init_window(1920, 1080)

	if cfg.load_snapshot:
		testbed.load_snapshot(cfg.load_snapshot_path)

	if cfg.optimize_extrinsics:
		testbed.nerf.training.optimize_extrinsics = True

	# testbed.shall_train = cfg.train if cfg.gui else True
	testbed.shall_train = True
	testbed.nerf.render_with_lens_distortion = True

	testbed.background_color = [0.0, 0.0, 0.0, 0.0]

	print("Default_color", testbed.background_color)
	print("Default_random_bg_color", testbed.nerf.training.random_bg_color)
	print("NeRF training ray near_distance ", cfg.near_distance)
	testbed.nerf.training.near_distance = cfg.near_distance

	old_training_step = 0
	n_steps = cfg.n_steps
	stable_steps = 0

	# If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
	# don't train by default and instead assume that the goal is to render screenshots,
	# compute PSNR, or render a video.
	if n_steps < 0 and (not cfg.load_snapshot or cfg.gui):
		n_steps = 35000

	if testbed.training_step is not 0:
		# if we loaded a snapshot, still train the model for a few steps
		n_steps += testbed.training_step
		old_training_step = testbed.training_step - 1

	tqdm_last_update = 0
	stable_steps_thresh = 50
	stable_loss_thresh = 0.0002
	max_infinity_steps = 40000
	if n_steps > 0:
		with tqdm(desc="Training", total=cfg.n_steps, unit="steps") as t:
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)

				# What will happen when training is done?
				if stable_steps > stable_steps_thresh or testbed.training_step > max_infinity_steps:
					if cfg.gui:
						testbed.shall_train = False
						break
					else:
						break

				if testbed.training_step >= n_steps:
					# If currently stable and not high loss, then break.
					if (stable_steps > stable_steps_thresh and testbed.loss < stable_loss_thresh) or not dynamic_time_extension or render_distract:
						if cfg.gui:
							testbed.shall_train = False
							break
						else:
							break
					# Else, make stable thresholds less strict and keep trying.
					else:
						stable_loss_thresh *= 1.5
						stable_steps_thresh /= 1.5
						n_steps *= 1.3
						stable_steps_thresh = int(stable_steps_thresh)
						n_steps = int(n_steps)

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if now - tqdm_last_update > 0.1:
					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now
					if testbed.shall_train: # Don't count stable_steps if paused.
						if testbed.loss < stable_loss_thresh:
							stable_steps += 1
						else:
							stable_steps = 0

	if cfg.save_snapshot:
		testbed.save_snapshot(cfg.save_snapshot_path, False)

	# Get optimised camera extrinsics from training view
	if cfg.optimize_extrinsics:
		opt_cam_poses = get_optimised_poses(cfg, testbed)
		np.save(os.path.join(cfg.data_dir, 'opt_cam_poses.npy'), opt_cam_poses)
		# free the GPU memory
		ngp.free_temporary_memory()
	else:
		opt_cam_poses = None

	# if cfg.save_mesh:
	# 	res = cfg.marching_cubes_res or 256
	# 	print(f"Generating mesh via marching cubes and saving to {cfg.save_mesh}. Resolution=[{res},{res},{res}]")
	# 	testbed.compute_and_save_marching_cubes_mesh(cfg.save_mesh, [res, res, res])

	return testbed, opt_cam_poses


def get_optimised_poses(cfg, testbed):
	fg_output_file = os.path.join(cfg.data_dir, "fg_transforms.json")
	bg_output_file = os.path.join(cfg.data_dir, "bg_transforms.json")

	# Save optimised camera extrinsics
	fg_out = {
		"fl_x": cfg.fx,
		"fl_y": cfg.fy,
		"k1": cfg.k1,
		"k2": cfg.k2,
		"k3": cfg.k3,
		"k4": cfg.k4,
		"p1": cfg.p1,
		"p2": cfg.p2,
		"is_fisheye": cfg.is_fisheye,
		"cx": cfg.cx,
		"cy": cfg.cy,
		"w": cfg.W,
		"h": cfg.H,
		"aabb_scale": 1,
		"scale": cfg.scale,
		"offset": cfg.offset,  # BRG (xyz)
		"frames": []
	}

	bg_out = {
		"fl_x": cfg.fx,
		"fl_y": cfg.fy,
		"k1": cfg.k1,
		"k2": cfg.k2,
		"k3": cfg.k3,
		"k4": cfg.k4,
		"p1": cfg.p1,
		"p2": cfg.p2,
		"is_fisheye": cfg.is_fisheye,
		"cx": cfg.cx,
		"cy": cfg.cy,
		"w": cfg.W,
		"h": cfg.H,
		"aabb_scale": 1,
		"scale": cfg.scale,
		"offset": cfg.offset,  # BRG (xyz)
		"frames": []
	}

	if cfg.camera_angle_x is not None:
		fg_out["camera_angle_x"] = cfg.camera_angle_x
		fg_out["camera_angle_y"] = cfg.camera_angle_y
		bg_out["camera_angle_x"] = cfg.camera_angle_x
		bg_out["camera_angle_y"] = cfg.camera_angle_y

	# Get optimised camera extrinsics from training view
	opt_cam_poses = []
	with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Saving RGBA images") as t:
		for i in t:
			matrix_ngp = testbed.nerf.training.get_camera_extrinsics(i)
			matrix_ngp = np.row_stack((matrix_ngp, np.array([0, 0, 0, 1])))
			# flip y and z axis to match the camera coordinate system in open3d
			matrix_accio = matrix_ngp.copy()
			matrix_accio[:3, 1] *= -1
			matrix_accio[:3, 2] *= -1
			opt_cam_poses.append(matrix_accio)

			# save to cache
			fg_name = "./images_fg/rgb_%04d.png" % i
			bg_name = "./images_bg/rgb_%04d.png" % i
			fg_frame = {"file_path": fg_name, "transform_matrix": matrix_ngp.tolist()}
			bg_frame = {"file_path": bg_name, "transform_matrix": matrix_ngp.tolist()}
			fg_out["frames"].append(fg_frame)
			bg_out["frames"].append(bg_frame)

	with open(fg_output_file, "w") as fg_output:
		json.dump(fg_out, fg_output, indent=2)

	with open(bg_output_file, "w") as bg_output:
		json.dump(bg_out, bg_output, indent=2)

	return opt_cam_poses


if __name__ == "__main__":
	config_file = "./configs/full_scene.json"
	cfg = Config(config_file)
	build_vis_model(cfg)
