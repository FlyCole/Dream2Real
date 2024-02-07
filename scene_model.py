import os
import pdb
import cv2
import numpy as np
import torch
from tqdm import tqdm

from vision_3d.pcd_visual_model import get_vis_pcds
from reconstruction.ngp_visual_model import get_vis_ngps
from vision_3d.physics_utils import get_phys_models
from segmentation.sam_seg import remove_components_at_edges

class ObjectModel():
    def __init__(self, name, vis_model, phys_model, init_pose, thumbnail, mask_idx):
        self.name = name
        self.vis_model = vis_model
        self.phys_model = phys_model
        self.pose = init_pose # T_world_obj
        self.thumbnail = thumbnail
        self.mask_idx = mask_idx

    def update_pose(self, new_pose):
        self.pose = new_pose

# NOTE: RGBs are in HWC format.
class SceneModel():
    def __init__(self, scene_centre, objs, bground_obj, rgbs, depths, opt_cam_poses,
                 intrinsics, masks, scene_bounds, scene_type, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # NOTE: bground also included in objs.
        self.objs = objs
        self.bground_obj = bground_obj
        self.scene_centre = scene_centre
        self.device = device
        self.rgbs = rgbs
        self.depths = depths
        self.opt_cam_poses = opt_cam_poses
        self.intrinsics = intrinsics
        self.masks = masks
        self.scene_bounds = scene_bounds
        self.scene_type = scene_type

class TaskModel():
    # task_bground_obj is a dummy ObjectModel which stores the visual model of the scene background specific to this task.
    def __init__(self, user_instr, goal_caption, norm_captions, scene_model, movable_obj, task_bground_obj, task_bground_masks, topdown):
        self.user_instr = user_instr
        self.goal_caption = goal_caption
        self.norm_captions = norm_captions
        self.scene_model = scene_model
        self.movable_obj = movable_obj
        self.task_bground_obj = task_bground_obj
        self.task_bground_masks = task_bground_masks
        self.movable_masks = torch.logical_not(scene_model.masks == movable_obj.mask_idx)
        self.topdown = topdown

    # Need to: create training img dataset where each image has following pixels masked out:
    #   Any obj not in relevant_objs (i.e. mask out distractors), and also movable_obj
    def create_task_bground_obj(scene_model, movable_obj, relevant_objs, out_scene_bound_masks, save_dir,
                                use_vis_pcds=False, pcds_type=None, single_view_idx=0, render_distractors=False,
                                use_cache=False, data_dir=None):
        if use_cache:
            print('Using cached visual model for task background')
        # Note that in task_bground_mask, we want the 0 index to be the task bground, and 1 to be anything else.
        # i.e. the only pixels which should be 1 are those which belong to movable or to distractor (not relevant) objs.
        task_bground_masks = torch.zeros_like(scene_model.masks)
        for obj in scene_model.objs:
            # Encourage movable object, distractors, and background to be transparent.
            if render_distractors: # Only mask movable object.
                if obj is movable_obj:
                    task_bground_masks[scene_model.masks == obj.mask_idx] = 1
            else:
                if obj is movable_obj or obj is scene_model.bground_obj or (obj not in relevant_objs):
                # if obj is movable_obj: # If you want bground to have everything except movable
                    task_bground_masks[scene_model.masks == obj.mask_idx] = 1

        # integrate out_of_scene_masks into task_bground_masks
        for i, mask in enumerate(out_scene_bound_masks):
            task_bground_masks[i] |= mask.cpu().bool().numpy()

        if use_vis_pcds:
            vis_model = get_vis_pcds(scene_model.rgbs, scene_model.depths, scene_model.opt_cam_poses, scene_model.intrinsics,
                                     task_bground_masks, 1, scene_model.scene_bounds, save_dir=save_dir, vis=False,
                                     use_cache=use_cache, pcds_type=pcds_type, single_view_idx=single_view_idx)[0]
        else:
            vis_model = get_vis_ngps(scene_model.rgbs, task_bground_masks, scene_model.scene_type,
                                     use_cache=use_cache, data_dir=data_dir, fg=False, render_distract=render_distractors)

        name = "__task_bground__"
        phys_model = None
        pose = torch.eye(4)
        thumbnail = None
        mask_idx = None # See note at top of method for proper mask idx convention regarding task bground.
        task_bground_obj = ObjectModel(name, vis_model, phys_model, pose, thumbnail, mask_idx)
        return task_bground_obj, task_bground_masks

    def create_movable_vis_model(scene_model, movable_obj, out_scene_bound_masks, save_dir,
                                 use_vis_pcds=False, pcds_type=None, single_view_idx=0,
                                 use_cache=False, data_dir=None):
        if use_cache:
            print('Using cached visual model for movable object')
        # The "not" is because we want idx 0 in mask to be movable obj pixels, and other idxs will be ignored since num_objs = 1.
        movable_masks = torch.logical_not(scene_model.masks == movable_obj.mask_idx)
        if use_vis_pcds:
            vis_model = get_vis_pcds(scene_model.rgbs, scene_model.depths, scene_model.opt_cam_poses, scene_model.intrinsics,
                                     movable_masks, 1, scene_model.scene_bounds, save_dir=save_dir, vis=False,
                                     use_cache=use_cache, pcds_type=pcds_type, single_view_idx=single_view_idx)[0]
        else:
            vis_model = get_vis_ngps(scene_model.rgbs, movable_masks, scene_model.scene_type,
                                     use_cache=use_cache, data_dir=data_dir, fg=True)

        return vis_model

    # Creates two physics models: movable obj and everything else (including distractors).
    def create_lazy_phys_mods(scene_model, movable_obj, scene_bounds, save_dir,
                              embodied=False,vis=False, use_cache=False, use_phys_tsdf=True,
                              use_vis_pcds=False, single_view_idx=0):
        fg_bg_masks = torch.where(scene_model.masks == movable_obj.mask_idx, torch.tensor(1), torch.tensor(0)) # So that 0 is background, 1 is foreground.
        num_objs = 2 # Foreground and background.
        [bground_phys, movable_phys], [bground_init_pose, movable_init_pose] = get_phys_models(scene_model.depths, scene_model.opt_cam_poses, scene_model.intrinsics, fg_bg_masks,
                                                                                               num_objs=num_objs, scene_bounds=scene_bounds, embodied=embodied, save_dir=save_dir,
                                                                                               vis=vis,use_cache=use_cache, use_phys_tsdf=use_phys_tsdf, use_vis_pcds=use_vis_pcds,
                                                                                               single_view_idx=single_view_idx)
        return [bground_phys, movable_phys], [bground_init_pose, movable_init_pose]

    def free_visual_models(self):
        self.task_bground_obj.vis_model = None
        # self.movable_obj.vis_model = None
        torch.cuda.empty_cache()

class ObjectModel2D():
    # masks should be same size as original image.
    # Yes mask may be redundant, but it makes code less complex.
    # obj_crop has shape (H, W, 4).
    def __init__(self, name, obj_crop, thumbnail, pos, mask):
        self.name = name
        self.obj_crop = obj_crop
        self.thumbnail = thumbnail
        self.pos = pos
        self.mask = mask