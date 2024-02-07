import argparse
import os
import pdb
from termcolor import colored
from cfg import Config
from dream2real import ImaginationEngine

if __name__ == "__main__":
    # data_dir need only have depth/, images/, seg_images/, associate_index.txt, scene_name.txt, poses.txt,  and transforms.json.
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the data directory containing raw scanning data. Not in version control.")
    parser.add_argument("out_dir", type=str, help="Path to directory containing intermediate output files e.g. segmentations. Not in version control.")
    parser.add_argument("cfg_path", type=str, help="Path to config file. Pass path to different config to run different variants/ablations. Is in version control.")
    parser.add_argument("user_instr", type=str, help="User instruction")
    parser.add_argument("--goal_caption", type=str, default=None, required=False, help="Goal caption (optional, by default inferred from user_instr)")
    parser.add_argument("--norm_captions", type=str, nargs='+', default=None, required=False, help="Normalising captions (optional, by default inferred from user_instr)")
    args = parser.parse_args()
    user_instr = args.user_instr
    goal_caption = args.goal_caption
    norm_captions = args.norm_captions

    # out_dir, not data_dir. We don't want to write anything to data_dir.
    cfg = Config(args.cfg_path, args.out_dir)

    assert not ((not cfg.use_cache_cam_poses) and cfg.use_cache_phys), "Cannot use new camera poses with old cached physics models. Disable use_cache_phys."
    assert not ((not cfg.use_cache_cam_poses) and cfg.use_cache_vis), "Cannot use new camera poses with old cached visual models. Disable use_cache_vis."
    assert not ((not cfg.use_cache_segs) and cfg.use_cache_captions), "Cannot use new segmentations with old cached captions. Disable use_cache_captions."
    # if cfg.use_cache_renders:
    #     assert os.path.exists(os.path.join(args.out_dir, 'cb_render/')), "Cannot use cached renders since cb_render directory not yet created and renders not yet created. Disable use_cache_renders."

    if not cfg.use_cache_segs:
        print(colored("Warning:", "red"), " about to delete and regenerate everything from segmentations onwards. Press Ctrl+C to cancel, or Enter to continue.")
        # input()

    # Check that data_dir exists.
    if not os.path.isdir(args.data_dir):
        raise ValueError("data_dir does not exist.")

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # If data not there already, copy it to out_dir.
    if not os.path.isdir(os.path.join(args.out_dir, "images")):
        os.system(f'cp -r {args.data_dir}/* {args.out_dir}')

    # Run the method.
    print(f'Running with config: {args.cfg_path}')
    imagination = ImaginationEngine(cfg)
    imagination.build_scene_model()
    task_model = imagination.interpret_user_instr(user_instr, goal_caption=goal_caption, norm_captions=norm_captions)
    movable_best_pose = imagination.dream_best_pose(task_model)
    print(colored("Predicted pose for movable object:", "green"))
    print(movable_best_pose)
