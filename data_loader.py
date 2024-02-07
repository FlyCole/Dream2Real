import pdb
import imgviz
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from cfg import Config
import glob
from torchvision import transforms
import open3d as o3d
import time

from vis_utils import visimg


class d2r_dataloader:
    def __init__(self, cfg):
        self.root_dir = cfg.data_dir
        self.rgb_dir = os.path.join(self.root_dir, "images")
        self.depth_dir = os.path.join(self.root_dir, "depth")

        self.traj_file = os.path.join(self.root_dir, "poses.txt")

        self.size = None
        self.width = cfg.width
        self.height = cfg.height
        self.data_device = "cuda:0"

        self.rgb_data = None
        self.depth_data = None
        self.T_WC_data = None
        self.dynamic_masks = None

    # depth is returned in metres.
    # rgb output format is HWC
    def load_rgbds(self, show=False):
        T_WC = np.loadtxt(self.traj_file, delimiter=" ").reshape([-1, 4, 4])
        size = len(T_WC)
        self.size = size
        self.rgb_data = torch.empty(size, self.height, self.width, 3, dtype=torch.uint8, device=self.data_device)
        self.depth_data = torch.empty(size, self.height, self.width, dtype=torch.float16, device=self.data_device)
        self.T_WC_data = torch.empty(size, 4, 4, dtype=torch.float32, device=self.data_device)

        print("Loading RGBD data...")
        for idx in tqdm(range(size)):
            rgb_file = os.path.join(self.rgb_dir, "rgb_%04d.png" % idx)
            depth_file = os.path.join(self.depth_dir, "depth_%04d.png" % idx)

            color = cv2.imread(rgb_file).astype(np.uint8)

            if show:
                cv2.imshow("Data Loader", color)
                cv2.waitKey(1)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_file, -1).astype(np.float16) / 1000

            if color is None or depth is None:
                print(rgb_file)
                print(depth_file)
                raise ValueError

            self.rgb_data[idx] = torch.tensor(color)
            self.depth_data[idx] = torch.tensor(depth)
            self.T_WC_data[idx] = torch.tensor(T_WC[idx])

        return self.rgb_data, self.depth_data, self.T_WC_data

    def remove_background(self, intrinsics, scene_phys_bounds, use_cache=False):
        print("Generating dynamic masks for backgrond...")
        out_path = os.path.join(self.root_dir, "images")
        if use_cache:
            print("Loading cached dynamic masks...")
            self.dynamic_masks = torch.empty(self.size, self.height, self.width, dtype=torch.uint8, device=self.data_device)
            for idx in tqdm(range(self.size)):
                mask_file = os.path.join(out_path, "dynamic_mask_rgb_%04d.png" % idx)
                dynamic_mask = cv2.imread(mask_file, -1)
                self.dynamic_masks[idx] = torch.tensor(dynamic_mask, dtype=torch.uint8, device=self.data_device)
            return self.dynamic_masks

        scene_phys_bounds = scene_phys_bounds.copy()
        scene_phys_bounds[0][2] = -100
        self.dynamic_masks = torch.empty_like(self.depth_data, dtype=torch.uint8, device=self.data_device)
        for depth_idx in tqdm(range(len(self.depth_data))):
            depth_img = self.depth_data[depth_idx].cpu().numpy()
            # project depth to 3D points
            depth_o3d = o3d.geometry.Image((depth_img * 1000).astype(np.uint16))
            T_cw = np.linalg.inv(self.T_WC_data[depth_idx].cpu().numpy())
            o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, intrinsics)
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, o3d_intrinsics, T_cw, project_valid_depth_only=False)

            dynamic_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            pcd_array = np.asarray(pcd.points)
            map_0 = pcd_array[:, 2] > -0.40
            map_1 = pcd_array[:, 0] < scene_phys_bounds[0][0]
            map_2 = pcd_array[:, 0] > scene_phys_bounds[1][0]
            map_3 = pcd_array[:, 1] < scene_phys_bounds[0][1]
            map_4 = pcd_array[:, 1] > scene_phys_bounds[1][1]
            map_5 = pcd_array[:, 2] < scene_phys_bounds[0][2]
            map_6 = pcd_array[:, 2] > scene_phys_bounds[1][2]
            map = map_0 & (map_1 | map_2 | map_3 | map_4 | map_5 | map_6)
            map = np.reshape(map, (self.height, self.width))
            map &= (depth_img != 0)
            dynamic_mask[map] = 255

            dynamic_mask = cv2.dilate(dynamic_mask, np.ones((50, 50), np.uint8), iterations=1)
            dynamic_mask = cv2.erode(dynamic_mask, np.ones((50, 50), np.uint8), iterations=1)

            self.dynamic_masks[depth_idx] = torch.tensor(dynamic_mask, dtype=torch.uint8, device=self.data_device)

            out_file = os.path.join(out_path, "dynamic_mask_rgb_%04d.png" % depth_idx)
            cv2.imwrite(out_file, dynamic_mask)

            # Uncomment to visualise dynamic masks applied to RGB images for debugging.
            # debug_img = self.rgb_data[depth_idx].cpu().numpy() * np.logical_not(dynamic_mask[:, :, np.newaxis])
            # debug_img = debug_img.astype(np.uint8)
            # visimg(debug_img)
            # pdb.set_trace()

        return self.dynamic_masks


if __name__ == '__main__':
    config_file = "configs/full_scene.json"
    cfg = Config(config_file)
    dataloader = d2r_dataloader(cfg)
    a, b, c = dataloader.load_rgbds()
