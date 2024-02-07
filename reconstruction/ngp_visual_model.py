import json

from data_loader import d2r_dataloader
from cfg import Config
from segmentation.XMem_infer import XMem_inference
from reconstruction.train_ngp import build_vis_model
import torch
import os
import numpy as np
import cv2
import sys
sys.path.append('./reconstruction/instant-ngp/build')
sys.path.append('./reconstruction/instant-ngp/scripts')
import pyngp as ngp
import pathlib
from tqdm import tqdm


# Create foreground and background task models.
def get_vis_ngps(rgbs, movable_masks, scene_type, use_cache=False, data_dir=None, fg=True, render_distract=False):
    if use_cache:
        # load cached ngp models.
        print('Using cached fg model for movable object')
        testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
        if fg:
            testbed.load_snapshot(os.path.join(data_dir, 'fg_base.ingp'))
        else:
            testbed.load_snapshot(os.path.join(data_dir, 'bg_base.ingp'))
        return testbed
    else:
        # rgba images generation
        if fg:
            out_path = os.path.join(data_dir, 'images_fg')
        else:
            out_path = os.path.join(data_dir, 'images_bg')
        os.makedirs(out_path, exist_ok=True)

        print('Generating images for training task ngp models')
        for i in tqdm(range(rgbs.shape[0])):
            rgb = rgbs[i].cpu().numpy()
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            mask = movable_masks[i].cpu().numpy()
            mask = 1 - mask
            rgba = np.concatenate([rgb, 255 * mask[:, :, np.newaxis]], axis=2)
            rgba = rgba.astype(np.uint8)
            cv2.imwrite(os.path.join(out_path, 'rgb_%04d.png' % i), rgba)

        # train foreground and background task models
        print('Training task ngp models')
        curr_file_path = pathlib.Path(__file__).parent.parent.absolute()

        if fg:
            fg_cfg = Config(os.path.join(curr_file_path, 'configs/fg_scene.json'), data_dir=data_dir)
            ngp_model, _ = build_vis_model(fg_cfg)
        else:
            bg_cfg = Config(os.path.join(curr_file_path, 'configs/bg_scene.json'), data_dir=data_dir)
            ngp_model, _ = build_vis_model(bg_cfg, render_distract=render_distract)

        return ngp_model


if __name__ == '__main__':
    config_file = "./configs/full_scene.json"
    cfg = Config(config_file)
    dataloader = d2r_dataloader(cfg)
    rgbs, _, _ = dataloader.load_rgbds()

    segmentor = XMem_inference()
    masks = segmentor.segment(rgbs, None, cfg.data_dir, show=False, use_cache=True)
    masks = [torch.tensor(mask) for mask in masks]
    masks = [mask[:, :, 0] for mask in masks]
    masks = torch.stack(masks, dim=0)

    movable_masks = masks == 1
    cv2.imshow('movable_masks', movable_masks[0].cpu().numpy().astype(np.uint8) * 255)
    cv2.waitKey(1)
    get_vis_ngps(rgbs, movable_masks, use_cache=False, data_dir=cfg.data_dir, fg=False)
    movable_masks = torch.logical_not(movable_masks)
    get_vis_ngps(rgbs, movable_masks, use_cache=False, data_dir=cfg.data_dir)
