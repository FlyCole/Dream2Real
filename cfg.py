import json
import numpy as np
import os
import utils
import pathlib


class Config:
    def __init__(self, config_file, data_dir):
        # setting params
        with open(config_file) as json_file:
            config = json.load(json_file)

        # dataset setting
        self.data_dir = data_dir
        self.files = os.path.join(data_dir, config["dataset"]["files"])

        # engine setting
        if "engine" in config:
            self.inpaint_holes = config["engine"]["inpaint_holes"]
            self.caption = config["engine"]["caption"]
            self.visseg = config["engine"]["visseg"]
            self.render_distractors = config["engine"]["render_distractors"]
            self.spatial_smoothing = config["engine"]["spatial_smoothing"]
            self.physics_only = config["engine"]["physics_only"]
            self.use_vis_pcds = config["engine"]["use_vis_pcds"]
            if self.use_vis_pcds:
                # 0: single view, 1: multi view
                self.pcds_type = config["engine"]["pcds_type"]
            else:
                self.pcds_type = None
            if "single_view_idx" in config["engine"]:
                self.single_view_idx = config["engine"]["single_view_idx"]
            else:
                self.single_view_idx = 0
            self.use_cache_dynamic_masks = config["engine"]["use_cache_dynamic_masks"]
            self.use_cache_segs = config["engine"]["use_cache_segs"]
            self.use_cache_cam_poses = config["engine"]["use_cache_cam_poses"]
            self.use_cache_captions = config["engine"]["use_cache_captions"]
            self.use_cache_phys = config["engine"]["use_cache_phys"]
            self.use_cache_vis = config["engine"]["use_cache_vis"]
            self.use_cache_llm = config["engine"]["use_cache_llm"]
            self.use_cache_renders = config["engine"]["use_cache_renders"]
            self.use_cache_goal_pose = config["engine"]["use_cache_goal_pose"]
            self.use_phys = config["engine"]["use_phys"]
            self.use_phys_tsdf = config["engine"]["use_phys_tsdf"]
            self.lazy_phys_mods = config["engine"]["lazy_phys_mods"]
            self.multi_view_captions = config["engine"]["multi_view_captions"]
            self.scene_type = config["engine"]["scene_type"]
            self.sample_res = config["engine"]["sample_res"]
            self.scene_centre = config["engine"]["scene_centre"]
            self.scene_phys_bounds = config["engine"]["scene_phys_bounds"]
            self.render_cam_pose_idx = config["engine"]["render_cam_pose_idx"]

        # trainer setting
        self.train = config["trainer"]["train"]
        self.depth_scale = 1 / config["trainer"]["scale"]
        self.training_device = config["trainer"]["train_device"]
        self.data_device = config["trainer"]["data_device"]
        self.load_snapshot = config["trainer"]["load_snapshot"]
        if self.load_snapshot:
            self.load_snapshot_path = os.path.join(self.data_dir, config["trainer"]["load_snapshot_path"])
        self.save_snapshot = config["trainer"]["save_snapshot"]
        if self.save_snapshot:
            self.save_snapshot_path = os.path.join(self.data_dir, config["trainer"]["save_snapshot_path"])
        self.n_steps = config["trainer"]["n_steps"]
        self.near_distance = config["trainer"]["near_distance"]
        self.optimize_extrinsics = config["trainer"]["optimize_extrinsics"]

        # renderer setting
        self.max_depth = config["render"]["depth_range"][1]
        self.min_depth = config["render"]["depth_range"][0]

        # camera setting
        if "camera" in config:
            self.mh = config["camera"]["mh"]
            self.mw = config["camera"]["mw"]
            self.height = config["camera"]["h"]
            self.width = config["camera"]["w"]
            self.H = self.height - 2 * self.mh
            self.W = self.width - 2 * self.mw
            if "camera_angle_x" in config["camera"]:
                self.camera_angle_x = config["camera"]["camera_angle_x"]
                self.camera_angle_y = config["camera"]["camera_angle_y"]
            else:
                self.camera_angle_x = None
                self.camera_angle_y = None
            if "is_fisheye" in config["camera"]:
                self.is_fisheye = config["camera"]["is_fisheye"]
            if "fx" in config["camera"]:
                self.fx = config["camera"]["fx"]
                self.fy = config["camera"]["fy"]
                self.cx = config["camera"]["cx"] - self.mw
                self.cy = config["camera"]["cy"] - self.mh
            else:   # for scannet
                intrinsic = utils.load_matrix_from_txt(os.path.join(self.data_dir, "intrinsic/intrinsic_depth.txt"))
                self.fx = intrinsic[0, 0]
                self.fy = intrinsic[1, 1]
                self.cx = intrinsic[0, 2] - self.mw
                self.cy = intrinsic[1, 2] - self.mh
            if "distortion" in config["camera"]:
                self.distortion_array = np.array(config["camera"]["distortion"])
            elif "k1" in config["camera"]:
                self.k1 = config["camera"]["k1"]
                self.k2 = config["camera"]["k2"]
                self.k3 = config["camera"]["k3"]
                self.k4 = config["camera"]["k4"]
                self.p1 = config["camera"]["p1"]
                self.p2 = config["camera"]["p2"]
                self.distortion_array = np.array([self.k1, self.k2, self.p1, self.p2, self.k3, self.k4])
            else:
                self.distortion_array = None
            self.aabb_scale = config["camera"]["aabb_scale"]
            self.scale = config["camera"]["scale"]
            self.offset = config["camera"]["offset"]

        # visualiser
        self.gui = config["vis"]["gui"]

        if "robot" in config:
            self.robot_cfg = config["robot"]