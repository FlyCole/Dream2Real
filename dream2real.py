import copy
import os
from PIL import Image
import numpy as np
import torch
from caption import Captioner
from clip_scoring import optimise_pose_grid
from vis_utils import visimg, pastel_colors
from diffusers import StableDiffusionInpaintPipeline
from diffusion import inpaint
from torchvision.transforms.functional import pil_to_tensor
import pathlib
import sys
import open3d as o3d
import pybullet as p
import pybullet_planning as pp
import pdb
from segmentation.XMem_infer import XMem_inference
from vision_3d.geometry_utils import vis_cost_volume, vis_multiverse
from vision_3d.physics_utils import create_unsupcol_check, get_phys_models
from vision_3d.camera_info import INTRINSICS_REALSENSE_1280
from segmentation.sam_seg import get_thumbnail
from reconstruction.train_ngp import build_vis_model
from reconstruction.combined_rendering import renderer
from scene_model import ObjectModel, SceneModel, TaskModel
from vision_3d.pcd_visual_model import PointCloudRenderer, get_vis_pcds
from data_loader import d2r_dataloader
from cfg import Config
from termcolor import colored
import argparse

os.nice(1) # We're here to run fast, not to make friends.

curr_dir_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(curr_dir_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
egi_gpu = total_memory_gb > 20

from lang.llm import LangModel

class ImaginationEngine():
    """Imagination engine for generating task models from user instructions."""
    def __init__(self, cfg, embodied=False):
        self.embodied = embodied

        # Initialise configs
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.use_phys = cfg.use_phys
        self.use_phys_tsdf = cfg.use_phys_tsdf
        self.lazy_phys_mods = cfg.lazy_phys_mods
        self.multi_view_captions = cfg.multi_view_captions
        self.use_cache_dynamic_masks = cfg.use_cache_dynamic_masks
        self.use_cache_segs = cfg.use_cache_segs
        self.use_cache_captions = cfg.use_cache_captions
        self.use_cache_phys = cfg.use_cache_phys
        self.use_cache_cam_poses = cfg.use_cache_cam_poses
        self.use_cache_renders = cfg.use_cache_renders
        self.use_cache_goal_pose = cfg.use_cache_goal_pose
        self.render_distractors = cfg.render_distractors
        self.spatial_smoothing = cfg.spatial_smoothing
        self.use_cache_vis = cfg.use_cache_vis
        self.use_vis_pcds = cfg.use_vis_pcds
        self.pcds_type = cfg.pcds_type
        self.render_cam_pose_idx = cfg.render_cam_pose_idx
        self.scene_type = cfg.scene_type
        self.topdown = cfg.scene_type in [0, 3]
        self.physics_only = cfg.physics_only
        self.single_view_idx = cfg.single_view_idx # Defaults to 0 if not specified in cfg.

        # Initialise datasets
        self.depths_gt = None

        # Allocate models to GPUs.
        self.captioner_device = "cuda:0"

        self.scene_model = None
        self.segmentor = XMem_inference()
        self.caption = cfg.caption
        if cfg.caption:
            self.captioner = Captioner(topdown=self.topdown, device=self.captioner_device, read_cache=self.use_cache_captions, cache_path=os.path.join(self.data_dir, 'captions.json'))
        self.inpaint = cfg.inpaint_holes
        self.visseg = cfg.visseg
        if cfg.inpaint_holes:
            self.inpainter = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", requires_safety_checker=False).to("cuda")

        self.lang_model = LangModel(cache_path=os.path.join(curr_dir_path, 'lang/cache.json'), read_cache=cfg.use_cache_llm)

        self.renderer = None
        # self.renderer = PointCloudRenderer()

        assert cfg.scene_centre is not None
        assert cfg.scene_phys_bounds is not None
        assert cfg.sample_res is not None
        self.scene_centre = cfg.scene_centre
        self.scene_phys_bounds = cfg.scene_phys_bounds  # Format: x_min, y_min, z_min, x_max, y_max, z_max
        self.sample_res = cfg.sample_res

    def build_scene_model(self, raw_data=None):
        """Build scene model from raw data.

        Args:
            raw_data: raw RGBD data. If None, loads from data_dir.

        Returns:
            None

        """
        print('Building scene model...')
        intrinsics = INTRINSICS_REALSENSE_1280
        dataloader = d2r_dataloader(self.cfg)
        rgbs, depths, raw_cam_poses = dataloader.load_rgbds() if raw_data is None else raw_data
        self.out_scene_bound_masks = dataloader.remove_background(intrinsics, self.scene_phys_bounds, use_cache=self.use_cache_dynamic_masks) # 0 means inside scene, 1 means outside.

        self.depths_gt = [depths[render_idx] for render_idx in self.render_cam_pose_idx]
        self.depths_gt = torch.stack(self.depths_gt, dim=0)

        # Segment scene.
        video_path = os.path.join(self.data_dir, "seg_images")
        if os.path.exists(video_path):
            masks = self.segmentor.segment_associate(video_path,
                                                     depths,
                                                     dataloader.T_WC_data,
                                                     intrinsics,
                                                     self.data_dir,
                                                     self.out_scene_bound_masks,
                                                     self.scene_centre,
                                                     show=self.visseg,
                                                     use_cache=self.use_cache_segs,
                                                     debug=False)
        else:
            masks = self.segmentor.segment(rgbs, depths, self.data_dir, show=self.visseg, use_cache=self.use_cache_segs)
        self.segmentor.free()
        masks = [torch.tensor(mask) for mask in masks]
        masks = torch.stack(masks, dim=0)

        # Assume that bground has mask_idx = 0 and outside scene bounds has mask_idx = 255
        # 0-based indexing. Includes bground obj but excludes outside scene bounds.
        if 255 in torch.unique(masks):
            num_objs = torch.unique(masks).shape[0] - 1
        else:
            num_objs = torch.unique(masks).shape[0]

        if self.use_cache_cam_poses:
            print('Using cached optimised camera poses')
            opt_cam_poses = np.load(os.path.join(self.data_dir, 'opt_cam_poses.npy'))
        else:
            _, opt_cam_poses = build_vis_model(self.cfg, dynamic_time_extension=False, render_distract=self.render_distractors)
        opt_cam_poses = [torch.tensor(pose) for pose in opt_cam_poses]

        if self.lazy_phys_mods:
            phys_models = [None] * num_objs
            init_poses = [None] * num_objs
        else:
            phys_models, init_poses = get_phys_models(depths, opt_cam_poses, intrinsics, masks, num_objs, self.scene_phys_bounds,
                                                    save_dir=os.path.join(self.data_dir, 'phys_mods/'),
                                                    vis=not self.use_cache_phys, use_cache=self.use_cache_phys,
                                                    use_phys_tsdf=self.use_phys_tsdf)

        captions, thumbnails = self.captioner.caption_objs(num_objs, rgbs, masks, self.lang_model, self.out_scene_bound_masks,
                                                           topdown=self.topdown, multi_view=self.multi_view_captions,
                                                           single_view_idx=self.single_view_idx)
        self.captioner.free()

        # Visual models are created lazily once task is known.
        vis_models = [None] * num_objs

        objs = []
        for obj_idx in range(num_objs):
            mask_idx = obj_idx
            obj = ObjectModel(captions[obj_idx], vis_models[obj_idx], phys_models[obj_idx], init_poses[obj_idx], thumbnails[obj_idx], mask_idx)
            objs.append(obj)

        self.scene_model = SceneModel(self.scene_centre, objs, objs[0], rgbs, depths, opt_cam_poses,
                                      intrinsics, masks, self.scene_phys_bounds, self.scene_type, device=device)

    def determine_movable_obj(self, user_instr):
        """Determine which object is movable based on user instruction.

        Args:
            user_instr: user instruction string

        Returns:
            movable_obj: ObjectModel of movable object
            movable_idx: index of movable object in scene_model.objs

        """
        obj_captions = [obj.name for obj in self.scene_model.objs]
        movable_idx = self.lang_model.get_movable_obj_idx(user_instr, obj_captions)
        movable_obj = self.scene_model.objs[movable_idx]
        return movable_obj, movable_idx

    def determine_relevant_objs(self, norm_caption, movable_obj_idx):
        """Determine which objects are relevant to the task based on the normalised caption.

        Relevant means not a distractor. For example, relevant objects could include the plate where apple is to be
        placed. Even though plate not movable.

        Args:
            norm_caption: normalised caption string
            movable_obj_idx: index of movable object in scene_model.objs

        Returns:
            relevant_objs: list of relevant ObjectModels

        """
        obj_captions = [obj.name for obj in self.scene_model.objs]
        relevant_idxs = self.lang_model.get_relevant_obj_idxs(norm_caption, obj_captions, movable_obj_idx)
        if len(relevant_idxs) == 0:
            raise RuntimeError(f'Error: None of the captioned objects were determined to be relevant.')
        relevant_objs = [self.scene_model.objs[idx] for idx in relevant_idxs]
        return relevant_objs

    def interpret_user_instr(self, user_instr, goal_caption=None, norm_captions=None):
        """Interpret user instruction and return task model.

        User instruction is parsed to determine the movable object and the relevant objects. The goal caption and
        normalised caption are inferred from the user instruction if not provided (which is the default version).

        Args:
            user_instr: user instruction string
            goal_caption: goal caption string
            norm_captions: list of normalised caption strings

        Returns:
            task_model: TaskModel for the task

        """
        if self.scene_model is None:
            raise RuntimeError("Must call build_scene_model() first before receiving user instructions")

        if goal_caption is None:
            print('Attempting to infer goal caption and normalising caption automatically from user instruction...')
            goal_caption, norm_caption = self.lang_model.parse_instr(user_instr)
            print(colored('User instruction: ', 'green'), user_instr)
            print(colored('Goal caption: ', 'green'), goal_caption)
            print(colored('Normalised caption: ', 'green'), norm_caption)
            norm_captions = [norm_caption]
        movable_obj, movable_obj_idx = self.determine_movable_obj(user_instr)
        relevant_objs = self.determine_relevant_objs(goal_caption, movable_obj_idx)

        # Create these before visual models since phys mods need memory during construction, but fine afterwards.
        if self.lazy_phys_mods:
            [bground_phys, movable_phys], [bground_init_pose, movable_init_pose] = TaskModel.create_lazy_phys_mods(self.scene_model, movable_obj, self.scene_phys_bounds,
                                                                                                                   save_dir=os.path.join(self.data_dir, 'phys_mod/'), embodied=self.embodied,
                                                                                                                   vis=False, use_cache=self.use_cache_phys, use_phys_tsdf=self.use_phys_tsdf,
                                                                                                                   use_vis_pcds=self.use_vis_pcds,
                                                                                                                   single_view_idx=self.single_view_idx)

        movable_obj.vis_model = TaskModel.create_movable_vis_model(self.scene_model,
                                                                   movable_obj,
                                                                   self.out_scene_bound_masks,
                                                                   os.path.join(self.data_dir, 'movable_vis_mod/'),
                                                                   use_vis_pcds=self.use_vis_pcds,
                                                                   pcds_type=self.pcds_type,
                                                                   single_view_idx=self.single_view_idx,
                                                                   use_cache=self.use_cache_vis,
                                                                   data_dir=self.data_dir)

        task_bground_obj, task_bground_masks = TaskModel.create_task_bground_obj(self.scene_model,
                                                                                 movable_obj,
                                                                                 relevant_objs,
                                                                                 self.out_scene_bound_masks,
                                                                                 os.path.join(self.data_dir, 'task_bground_vis_mod/'),
                                                                                 use_vis_pcds=self.use_vis_pcds,
                                                                                 pcds_type=self.pcds_type,
                                                                                 single_view_idx=self.single_view_idx,
                                                                                 render_distractors=self.render_distractors,
                                                                                 use_cache=self.use_cache_vis,
                                                                                 data_dir=self.data_dir)

        if self.lazy_phys_mods:
            movable_obj.phys_model = movable_phys
            movable_obj.pose = movable_init_pose
            task_bground_obj.phys_model = bground_phys

        task_model = TaskModel(user_instr, goal_caption, norm_captions, self.scene_model, movable_obj, task_bground_obj, task_bground_masks, self.topdown)
        return task_model

    def dream_best_pose(self, task_model, vis_cost_vol=True):
        """Dream best pose for movable object based on task model.

        Args:
            task_model: TaskModel for the task
            vis_cost_vol: whether to visualise the cost volume

        Returns:
            best_pose: best pose for movable object

        """
        movable_init_pose = task_model.movable_obj.pose
        # Defining validity checks.
        # Takes in list of check functions, and returns a function that composes them.
        def compose_checks(checks):
            def composed_check(pose_batch, task_model, valid_so_far):
                valid_so_far = valid_so_far.clone()
                for check in checks:
                    valid_so_far &= check(pose_batch, task_model, valid_so_far)
                return valid_so_far
            return composed_check

        if self.use_phys and not self.use_cache_renders:
            # Setting up physics checking simulator if required.
            # If embodied, should already have PyBullet instance running.
            pyb_planner = pp # Yes.
            if not self.embodied:
                # Do not use PyBullet GUI if using visual pcd renderer because they conflict. Can also close PyBullet temporarily for visual renders in that case.
                pyb_planner.connect(use_gui=False) # not self.use_vis_pcds) # OPT: do not use GUI to save resources.

            unsupcol_check, static_obj_handles, movable_handles = create_unsupcol_check(pyb_planner,
                                                                                        task_model,
                                                                                        self.sample_res,
                                                                                        self.embodied,
                                                                                        lazy_phys_mods=self.lazy_phys_mods)
            self.static_phys_handles = static_obj_handles # Used by robot during motion planning.
            self.movable_phys_handle = movable_handles[0]
            shutdown_pyb = lambda pose_batch, task_model, valid_so_far: valid_so_far if pyb_planner.disconnect() else valid_so_far # For the one-liner.
            if self.embodied:
                phys_check = unsupcol_check
            else:
                phys_check = compose_checks([unsupcol_check, shutdown_pyb]) # Isn't it beautiful?
        else:
            all_valid = lambda pose_batch, task_model, valid_so_far: torch.ones(len(pose_batch), dtype=torch.bool)
            phys_check = all_valid

        # Choose renderer.
        if self.use_vis_pcds and not self.use_cache_goal_pose:
            self.renderer = PointCloudRenderer()
        else:
            self.renderer = renderer(self.data_dir, task_model)

        # Choose final best pose.
        if self.use_cache_goal_pose:
            best_pose = torch.tensor(np.loadtxt(os.path.join(self.data_dir, 'goal_pose.txt'))).float().to(device)
            pose_batch = torch.tensor(np.loadtxt(os.path.join(self.data_dir, 'pose_batch.txt'))).float().to(device)
            pose_scores = torch.tensor(np.loadtxt(os.path.join(self.data_dir, 'pose_scores.txt'))).float().to(device)
            # visualise best render
            best_render = Image.open(os.path.join(self.data_dir, 'best_render.png'))
            best_render.show()
        else:
            best_pose, pose_batch, pose_scores = optimise_pose_grid(self.renderer,
                                                                      self.depths_gt,
                                                                      self.render_cam_pose_idx,
                                                                      task_model,
                                                                      self.data_dir,
                                                                      sample_res=self.sample_res,
                                                                      phys_check=phys_check,
                                                                      use_templates=False,
                                                                      scene_type=self.scene_type,
                                                                      use_vis_pcds=self.use_vis_pcds,
                                                                      use_cache_renders=self.use_cache_renders,
                                                                      smoothing=self.spatial_smoothing,
                                                                      physics_only=self.physics_only)
            np.savetxt(os.path.join(self.data_dir, 'goal_pose.txt'), best_pose.cpu().numpy())
            np.savetxt(os.path.join(self.data_dir, 'pose_batch.txt'), pose_batch.cpu().numpy())
            np.savetxt(os.path.join(self.data_dir, 'pose_scores.txt'), pose_scores.cpu().numpy())

        # Visualise best pose.
        tsdf_vis = True # Else: VHACD.
        if vis_cost_vol and (self.use_cache_goal_pose or pose_scores is not None):
            if self.use_vis_pcds:
                bground_vis = task_model.task_bground_obj.vis_model
                bground_geoms = [bground_vis]
            elif self.lazy_phys_mods:
                if tsdf_vis:
                    bground_geoms = [os.path.join(self.data_dir, 'phys_mod/mesh_concave_0.obj')]
                else:
                    bground_geoms = [task_model.task_bground_obj.phys_model]
            else:
                bground_geoms = [obj.phys_model for obj in task_model.scene_model.objs if obj is not task_model.movable_obj]

            if tsdf_vis:
                movable_geom = os.path.join(self.data_dir, 'phys_mod/mesh_concave_1.obj')
            else:
                movable_geom = task_model.movable_obj.phys_model
            if not self.use_vis_pcds:
                vis_cost_volume(pose_scores, self.sample_res, pose_batch, bground_geoms)
                if not tsdf_vis:
                    vis_multiverse(pose_scores, self.sample_res, pose_batch, bground_geoms, movable_geom, movable_init_pose)
                best_ori = best_pose.view(4, 4)[:3, :3].cpu().numpy()

            pose_transform = best_pose.cpu() @ movable_init_pose.inverse().cpu()
            if self.use_vis_pcds:
                pass
                # vis_movable_pcd = copy.deepcopy(task_model.movable_obj.vis_model)
                # vis_movable_pcd.transform(pose_transform.numpy())
                # if not vis_movable_pcd.is_empty():
                #     o3d.visualization.draw_geometries([bground_vis, vis_movable_pcd])
            else:
                meshes = [o3d.io.read_triangle_mesh(phys_model) for phys_model in bground_geoms]
                meshes.append(o3d.io.read_triangle_mesh(movable_geom))
                for i, mesh in enumerate(meshes):
                    mesh.compute_vertex_normals()
                    col = pastel_colors[i % pastel_colors.shape[0]] / 255.0
                    mesh.paint_uniform_color(col)
                    if i == len(meshes) - 1:
                        mesh.transform(pose_transform.numpy())
                o3d.visualization.draw_geometries(meshes)

        return best_pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, help="Path to the .json config file")
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    parser.add_argument("user_instr", type=str, help="User instruction")
    parser.add_argument("--goal_caption", type=str, default=None, required=False, help="Goal caption (optional, by default inferred from user_instr)")
    parser.add_argument("--norm_captions", type=str, nargs='+', default=None, required=False, help="Normalising captions (optional, by default inferred from user_instr)")
    args = parser.parse_args()
    user_instr = args.user_instr
    goal_caption = args.goal_caption
    norm_captions = args.norm_captions

    cfg = Config(args.cfg_path, args.data_dir)
    if not os.path.exists(args.data_dir):
        raise ValueError("The specified data_dir does not exist.")

    assert not ((not cfg.use_cache_cam_poses) and cfg.use_cache_phys), "Cannot use new camera poses with old cached physics models. Disable use_cache_phys."
    assert not ((not cfg.use_cache_cam_poses) and cfg.use_cache_vis), "Cannot use new camera poses with old cached visual models. Disable use_cache_vis."
    assert not ((not cfg.use_cache_segs) and cfg.use_cache_captions), "Cannot use new segmentations with old cached captions. Disable use_cache_captions."
    if cfg.use_cache_renders:
        assert os.path.exists(os.path.join(args.data_dir, 'cb_render/')), "Cannot use cached renders since cb_render directory not yet created and renders not yet created. Disable use_cache_renders."

    if not egi_gpu:
        caption = False
        print(colored("Warning:", "red"), " setting caption to False automatically based on GPU availability")

    if not cfg.use_cache_segs:
        print(colored("Warning:", "red"), " about to delete and regenerate everything from segmentations onwards. Press Ctrl+C to cancel, or Enter to continue.")
        input()

    imagination = ImaginationEngine(cfg)

    imagination.build_scene_model()

    if goal_caption is not None:
        print('Using goal caption: ', goal_caption)
        print('Using normalising captions: ', norm_captions)
    task_model = imagination.interpret_user_instr(user_instr, goal_caption=goal_caption, norm_captions=norm_captions)
    movable_best_pose = imagination.dream_best_pose(task_model)
    print(colored("Predicted pose for movable object:", "green"))
    print(movable_best_pose)
