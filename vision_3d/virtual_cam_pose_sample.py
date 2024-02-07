import numpy as np
from scipy.spatial.transform import Rotation as R
import pdb

def get_virtual_cam_poses(task_model, render_cam_pose_idx):
    # concatenate all cam poses
    cam_poses = np.concatenate([task_model.scene_model.opt_cam_poses[idx].cpu().unsqueeze(0).numpy()
                                for idx in render_cam_pose_idx], axis=0)
    return cam_poses