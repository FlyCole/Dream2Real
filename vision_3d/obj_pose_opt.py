import pdb
import torch
from pytorch3d.transforms import euler_angles_to_matrix

# NOTE: these are absolute poses which are being sampled (i.e. in world frame).
# Not transforms relative to init pose.
# Return shape: (batch_size, 16 because flattened homogeneous matrix)
def sample_poses_grid(task_model, sample_res=[40, 40, 1, 1, 1, 1], scene_type=0):
    scene_model = task_model.scene_model
    device = scene_model.device

    # NOTE: this essentially defines the scene bounds for each scene.
    # In future, can be automated using depth measurements to get scene bounds.
    # Of course, these bounds are the same for each task within each scene.

    if scene_type == 0: # Pool table:
        x_lo, x_hi = torch.tensor([-0.12, 0.04]) + scene_model.scene_centre[0]
        y_lo, y_hi = torch.tensor([-0.10, 0.06]) + scene_model.scene_centre[1]
        z_lo, z_hi = torch.tensor([0.00, 0.085]) + scene_model.scene_centre[2]
        x_ori_lo, x_ori_hi = torch.tensor([0, 0])
        y_ori_lo, y_ori_hi = torch.tensor([0, 0])
        z_ori_lo, z_ori_hi = torch.tensor([0, 0])
    elif scene_type == 1: # Shelf:
        x_lo, x_hi = torch.tensor([-0.15, 0.20]) + scene_model.scene_centre[0]
        y_lo, y_hi = torch.tensor([0.40, 0.44]) + scene_model.scene_centre[1]
        z_lo, z_hi = torch.tensor([0.04, 0.41]) + scene_model.scene_centre[2]
        x_ori_lo, x_ori_hi = torch.tensor([-torch.pi, torch.pi / 2])
        y_ori_lo, y_ori_hi = torch.tensor([-torch.pi, torch.pi / 2])
        z_ori_lo, z_ori_hi = torch.tensor([-torch.pi, torch.pi / 2])
    elif scene_type == 3: # Shopping:
        x_lo, x_hi = torch.tensor([-0.19, 0.15]) + scene_model.scene_centre[0]
        y_lo, y_hi = torch.tensor([-0.25, 0.10]) + scene_model.scene_centre[1]
        z_lo, z_hi = torch.tensor([0.00, 0.14]) + scene_model.scene_centre[2]
        x_ori_lo, x_ori_hi = torch.tensor([0, 0])
        y_ori_lo, y_ori_hi = torch.tensor([0, 0])
        z_ori_lo, z_ori_hi = torch.tensor([0, 0])
    else:
        raise NotImplementedError("scene_type %d not implemented" % scene_type)

    xs = torch.linspace(x_lo, x_hi, sample_res[0]).to(device)
    ys = torch.linspace(y_lo, y_hi, sample_res[1]).to(device)
    zs = torch.linspace(z_lo, z_hi, sample_res[2]).to(device)
    x_oris = torch.linspace(x_ori_lo, x_ori_hi, sample_res[3]).to(device)
    y_oris = torch.linspace(y_ori_lo, y_ori_hi, sample_res[4]).to(device)
    z_oris = torch.linspace(z_ori_lo, z_ori_hi, sample_res[5]).to(device)

    pose_combos = torch.cartesian_prod(xs, ys, zs, x_oris, y_oris, z_oris)
    pose_batch = torch.eye(4).repeat(pose_combos.shape[0], 1, 1).to(device)
    pose_batch[:, :3, 3] = pose_combos[:, :3]
    eulers = pose_combos[:, 3:]
    rot_mats = euler_angles_to_matrix(eulers, 'XYZ')
    pose_batch[:, :3, :3] = rot_mats
    pose_batch = pose_batch.reshape(-1, 16)

    return pose_batch