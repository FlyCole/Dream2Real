import contextlib
import os
import pdb
import time
import cv2

import numpy as np
import pybullet as p
import pybullet_planning as pp
import open3d as o3d
import open3d.core as o3c
import torch
from tqdm import tqdm
from vis_utils import visimg
from vis_utils import pastel_colors
from pytorch3d.transforms import matrix_to_quaternion

GRAVITY_DIRECTION = np.array([0, 0, -1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# num_objs includes bground.
# OPT: can downsample more and earlier if needed.
# NOTE: returns not meshes but mesh paths, because that is what PyBullet wants.
def get_phys_models(depths, cam_poses, intrinsics, masks, num_objs, scene_bounds, embodied=False,
                    save_dir=None, vis=False, use_cache=True, use_phys_tsdf=False, use_vis_pcds=False,
                    single_view_idx=0):
    if use_cache:
        print('Using cached physics models')
        init_poses = []
        mesh_paths = []
        if vis:
            init_meshes = []
        for obj_id in range(num_objs):
            mesh_path = os.path.join(save_dir, f'mesh_{obj_id}.obj')
            mesh_paths.append(mesh_path)
            if vis:
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                init_meshes.append(mesh)
            init_pose = np.loadtxt(f'{save_dir}/init_pose_{obj_id}.txt')
            init_pose = torch.tensor(init_pose).float()
            init_poses.append(init_pose)
        if vis:
            for obj_id in range(num_objs):
                mesh = init_meshes[obj_id]
                mesh.compute_vertex_normals()
                col = pastel_colors[obj_id % len(pastel_colors)] / 255.0
                mesh.paint_uniform_color(col)
            o3d.visualization.draw_geometries(init_meshes, mesh_show_back_face=True)
        return mesh_paths, init_poses

    print('Creating physics models...')
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    o3d_device = o3c.Device("CPU:0")
    init_poses = []
    if use_phys_tsdf:
        init_meshes = []
        for obj_id in range(num_objs):
            obj_vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'), attr_dtypes=(o3c.float32, o3c.float32), attr_channels=((1), (1)),
                                                    voxel_size=0.002, block_resolution=16, block_count=100000, device=o3d_device)
            print('Building TSDF...')
            if use_vis_pcds:
                frame_range = [single_view_idx] * 4
            else:
                frame_range = range(len(depths))
            for frame_id in tqdm(frame_range):
                depth = depths[frame_id].clone().cpu().numpy()
                cam_pose = cam_poses[frame_id].cpu().numpy()
                mask = masks[frame_id].clone()
                mask = mask == obj_id

                # Erode mask to counter outliers due to mask imperfections.
                # Only need to do this significantly for the background, to prevent essentially adding copy of fg obj to bg TSDF.
                # TSDF does reasonably good job of eliminating outliers in foreground obj case.
                if obj_id == 0:
                    erosion_kernel_size = 20
                else:
                    erosion_kernel_size = 8
                mask = mask.cpu().numpy().astype(np.uint8)
                kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1).astype(np.bool)

                depth[~mask] = 0
                height = depth.shape[0]
                width = depth.shape[1]
                depth = o3d.t.geometry.Image(o3c.Tensor((depth * 1000).astype(np.uint16)).to(o3d_device)).to(o3d_device)
                T_cw = o3c.Tensor(np.linalg.inv(cam_pose)).to(o3d_device)
                o3d_intrinsics = o3c.Tensor(intrinsics).to(o3d_device)

                try:
                    frustrum_block_coords = obj_vbg.compute_unique_block_coordinates(depth, o3d_intrinsics, T_cw, depth_scale=1000.0, depth_max=3.0)
                    obj_vbg.integrate(frustrum_block_coords, depth, o3d_intrinsics, T_cw, depth_scale=1000.0, depth_max=3.0)
                except RuntimeError as e:
                    # This can happen when the current frame does not contain the current object.
                    pass

            init_mesh = obj_vbg.extract_triangle_mesh()
            init_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(init_mesh.vertex.positions.cpu().numpy()), o3d.utility.Vector3iVector(init_mesh.triangle.indices.cpu().numpy()))

            crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=scene_bounds[0], max_bound=scene_bounds[1])
            init_mesh = init_mesh.crop(crop_bbox)

            # Remove very small disconnected components (due to depth noise and mask errors).
            triangle_clusters, cluster_n_triangles, _ = init_mesh.cluster_connected_triangles()
            keep_threshold = 0.02 * np.max(cluster_n_triangles)
            removal_mask = [cluster_n_triangles[triangle_clusters[i]] < keep_threshold for i in range(len(triangle_clusters))]
            init_mesh.remove_triangles_by_mask(removal_mask)

            init_meshes.append(init_mesh)

            init_pose = torch.eye(4)
            init_pose[:3, 3] = torch.tensor(init_mesh.get_center())
            init_poses.append(init_pose)

            if save_dir is not None:
                init_pose_out_path = os.path.join(save_dir, f'init_pose_{obj_id}.txt')
                np.savetxt(init_pose_out_path, init_pose.numpy())

    else:
        crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=scene_bounds[0], max_bound=scene_bounds[1])
        frame_voxel_size = 0.002
        obj_voxel_size = 0.002
        outlier_neighbours = 30
        outlier_std_ratio = 1.05
        obj_pcds = []
        for obj_id in range(num_objs):
            obj_pcd = o3d.geometry.PointCloud()
            for frame_id in range(len(depths)):
                depth = depths[frame_id].clone().cpu().numpy()
                cam_pose = cam_poses[frame_id].cpu().numpy()
                mask = masks[frame_id].clone()
                mask = mask == obj_id

                # Erode mask to counter outliers due to mask imperfections.
                mask = mask.cpu().numpy().astype(np.uint8)
                kernel = np.ones((15, 15), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1).astype(np.bool)

                depth[~mask] = 0
                height = depth.shape[0]
                width = depth.shape[1]
                depth = o3d.geometry.Image((depth * 1000).astype(np.uint16))
                T_cw = np.linalg.inv(cam_pose)
                o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsics)
                frame_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, o3d_intrinsics, T_cw, depth_trunc=1000, depth_scale=1000)
                frame_pcd = frame_pcd.crop(crop_bbox)
                frame_pcd = frame_pcd.voxel_down_sample(frame_voxel_size)
                obj_pcd += frame_pcd

            _, inlier_idxs = obj_pcd.remove_statistical_outlier(nb_neighbors=outlier_neighbours, std_ratio=outlier_std_ratio)
            obj_pcd = obj_pcd.select_by_index(inlier_idxs)
            obj_pcd = obj_pcd.voxel_down_sample(obj_voxel_size)
            obj_pcds.append(obj_pcd)

            init_pose = torch.eye(4)
            init_pose[:3, 3] = torch.tensor(obj_pcd.get_center())
            init_poses.append(init_pose)

            if save_dir is not None:
                pcd_out_path = os.path.join(save_dir, f'obj_{obj_id}.pcd')
                o3d.io.write_point_cloud(pcd_out_path, obj_pcd)
                init_pose_out_path = os.path.join(save_dir, f'init_pose_{obj_id}.txt')
                np.savetxt(init_pose_out_path, init_pose.numpy())

        init_meshes = [create_mesh(pcd) for pcd in obj_pcds]

    if not embodied:
        p.connect(p.DIRECT)

    mesh_paths = []
    if save_dir is not None:
        for obj_id in range(num_objs):
            mesh_out_concave_path = os.path.join(save_dir, f'mesh_concave_{obj_id}.obj')
            o3d.io.write_triangle_mesh(mesh_out_concave_path, init_meshes[obj_id])
            mesh_out_convex_path = os.path.join(save_dir, f'mesh_{obj_id}.obj')

            # Get mesh size and rescale VHACD resolution appropriately.
            # max_bound = init_meshes[obj_id].get_max_bound()
            # min_bound = init_meshes[obj_id].get_min_bound()
            # diagonal_size = np.sqrt(((max_bound - min_bound) ** 2).sum())
            # vhacd_res = int(diagonal_size * 1000000)

            if obj_id == 0: # Assume background
                vhacd_res = 1000000
            else:
                vhacd_res = 10000

            # Redirect noisy vhacd output to devnull instead of printing to console.
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    p.vhacd(mesh_out_concave_path, mesh_out_convex_path, os.path.join(save_dir, f'mesh_vhacd_{obj_id}.log'), resolution=vhacd_res, depth=80, concavity=0.00002, gamma=0.00002, minVolumePerCH=0.00002, maxNumVerticesPerCH=64)
            mesh_paths.append(mesh_out_convex_path)

    if not embodied:
        p.disconnect()

    if vis:
        vis_meshes = []
        for obj_id in range(num_objs):
            mesh_path = os.path.join(save_dir, f'mesh_concave_{obj_id}.obj')
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh.compute_vertex_normals()
            col = pastel_colors[obj_id % pastel_colors.shape[0]] / 255.0
            mesh.paint_uniform_color(col)
            vis_meshes.append(mesh)
        o3d.visualization.draw_geometries(vis_meshes, mesh_show_back_face=True)

        vis_meshes = []
        for obj_id in range(num_objs):
            mesh_path = os.path.join(save_dir, f'mesh_{obj_id}.obj')
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh.compute_vertex_normals()
            col = pastel_colors[obj_id % pastel_colors.shape[0]] / 255.0
            mesh.paint_uniform_color(col)
            vis_meshes.append(mesh)
        o3d.visualization.draw_geometries(vis_meshes, mesh_show_back_face=True)

    # if vis:
        # for obj_id in range(num_objs):
        #     obj_pcd = obj_pcds[obj_id]
        #     col = pastel_colors[obj_id % len(pastel_colors)] / 255.0
        #     obj_pcd.paint_uniform_color(col)
        # o3d.visualization.draw_geometries(obj_pcds)

    print('Physics models created.')
    return mesh_paths, init_poses

# OPT: pyBullet has satCollision (separating axis theorem), maybe faster.
# OPT: maybe faster if not doing pairwise.
def create_unsupcol_check(pyb_planner, task_model, sample_res, embodied, unsup_thresh=0.02, lazy_phys_mods=True, stability_check=True):
    static_obj_handles = [] # Used later in pipeline for collision checking.
    movable_handles = [] # Used later in pipeline for collision checking.
    phys_obj_list = [task_model.task_bground_obj, task_model.movable_obj] if lazy_phys_mods else task_model.scene_model.objs
    for obj in phys_obj_list:
        scale = 1.0
        col_handle = p.createCollisionShape(p.GEOM_MESH, fileName=obj.phys_model, meshScale=[scale, scale, scale]) # flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        obj_handle = p.createMultiBody(baseCollisionShapeIndex=col_handle) # opt: use batchPositions
        if obj is task_model.movable_obj:
            # We only have one movable currently but this list is just to deal with function scope.
            movable_handle = obj_handle
            movable_handles.append(movable_handle)
        else:
            static_obj_handles.append(obj_handle)

    orig_movable_pos = p.getBasePositionAndOrientation(movable_handle)
    def unsupcol_check(pose_batch, task_model, valid_so_far, disallow_regrasp=embodied):
        valid_so_far = valid_so_far.clone()

        pose_batch = pose_batch.view(-1, 4, 4)
        init_pose = task_model.movable_obj.pose.to(device).repeat(pose_batch.shape[0], 1, 1)
        transforms = torch.matmul(pose_batch, torch.inverse(init_pose))
        pos_batch = transforms[:, :3, 3].cpu().numpy()
        quat_batch = matrix_to_quaternion(transforms[:, :3, :3]).cpu().numpy()
        quat_batch = quat_batch[:, [1, 2, 3, 0]] # Convert from wxyz to xyzw for PyBullet.

        # The idea is to remove duplicate orientations which have the same rotation matrix.
        # First work out which indices to remove for first pos, then repeat for all positions.
        sampled_oris_per_pos = sample_res[3] * sample_res[4] * sample_res[5]
        first_pos_oris = pose_batch[:sampled_oris_per_pos, :3, :3]
        first_pos_validity_mask = torch.ones(sampled_oris_per_pos, dtype=torch.bool).to(first_pos_oris.device)
        # OPT: batch.
        oris_seen_so_far = []
        for i in range(first_pos_oris.shape[0]):
            ori = first_pos_oris[i]
            seen = False
            for seen_ori in oris_seen_so_far:
                if torch.all(torch.isclose(ori, seen_ori, atol=0.01)):
                    seen = True
                    break
            if seen:
                first_pos_validity_mask[i] = 0
            else:
                oris_seen_so_far.append(ori)
        first_pos_validity_mask = first_pos_validity_mask.repeat(sample_res[0] * sample_res[1] * sample_res[2]) # Repeat mask for each position.
        valid_so_far &= first_pos_validity_mask
        print(f'Of {pose_batch.shape[0]} sampled poses, {first_pos_validity_mask.sum()} pass orientation uniqueness check.')

        # OPT: batch.
        first_pos_validity_mask = torch.ones(sampled_oris_per_pos, dtype=torch.bool).to(first_pos_oris.device)
        if disallow_regrasp:
            for i in range(first_pos_oris.shape[0]):
                if valid_so_far[i] == 0:
                    first_pos_validity_mask[i] = 0
                    continue
                ori = first_pos_oris[i]
                ori_facing_cam = False
                obj_z_vector = ori[:, 2]
                # Check if obj_z_vector is close to positive z axis vector.
                if obj_z_vector @ torch.tensor([0, 0, 1.0]).to(obj_z_vector.device) > 0.9:
                    ori_facing_cam = True
                # Also allowed to face negative y vector.
                if obj_z_vector @ torch.tensor([0, -1.0, 0]).to(obj_z_vector.device) > 0.9:
                    ori_facing_cam = True
                if not ori_facing_cam:
                    first_pos_validity_mask[i] = 0

        # Now apply same mask to all positions as before.
        first_pos_validity_mask = first_pos_validity_mask.repeat(sample_res[0] * sample_res[1] * sample_res[2]) # Repeat mask for each position.
        valid_so_far &= first_pos_validity_mask
        print(f'Of {pose_batch.shape[0]} sampled poses, {first_pos_validity_mask.sum()} also pass regrasp check.')

        # For debugging.
        all_poses_same = torch.all(torch.isclose(pose_batch, pose_batch[0].repeat(pose_batch.shape[0], 1, 1)))

        print("Checking each pose for colliding, unsupported or unstable objects...")
        for pose_idx in tqdm(range(pose_batch.shape[0])):
            if not valid_so_far[pose_idx]:
                continue

            pos = pos_batch[pose_idx]
            quat = quat_batch[pose_idx]

            # Check if object is in collision.
            pyb_planner.set_pose(movable_handle, (pos, quat))
            for other_obj in static_obj_handles:
                col = pyb_planner.pairwise_collision(movable_handle, other_obj)
                if col:
                    valid_so_far[pose_idx] = 0
                    break

            if not valid_so_far[pose_idx]:
                if all_poses_same:
                    print("Pose is in collision.")
                continue

            # Check if object is unsupported. Move object down: if not in collision, then unsupported.
            lower_pos = pos + unsup_thresh * GRAVITY_DIRECTION
            pyb_planner.set_pose(movable_handle, (lower_pos, quat))
            valid_so_far[pose_idx] = 0 # Will be set back to 1 if we can find a support for this obj.
            abs_lower_pos = pose_batch[pose_idx, :3, 3].cpu().numpy()
            below_table = abs_lower_pos[2] < task_model.scene_model.scene_centre[2]
            if below_table:
                valid_so_far[pose_idx] = 1
            else:
                for other_obj in static_obj_handles: # Check other objs as possible supports.
                    col = pyb_planner.pairwise_collision(movable_handle, other_obj)
                    if col:
                        valid_so_far[pose_idx] = 1
                        break

            if not valid_so_far[pose_idx]:
                if all_poses_same:
                    print("Pose is unsupported.")
                continue

            # Check if object is stable. If lower_pos is below table then already know that stable.
            # We know that colliding in lower pos. Should also be colliding in nearby lower poses.
            if stability_check and not below_table:
                p_dist = 0.04
                perturb_vecs = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0])]
                perturb_vecs = [p_dist * v for v in perturb_vecs]
                for perturb_vec in perturb_vecs:
                    perturbed_pos = lower_pos + perturb_vec
                    pyb_planner.set_pose(movable_handle, (perturbed_pos, quat))
                    any_col = False
                    for other_obj in static_obj_handles:
                        col = pyb_planner.pairwise_collision(movable_handle, other_obj)
                        if col:
                            any_col = True
                            break
                    if not any_col:
                        valid_so_far[pose_idx] = 0
                        break

            if not valid_so_far[pose_idx]:
                if all_poses_same:
                    print("Pose is unstable.")
                continue

        # Reset pose in sim.
        p.resetBasePositionAndOrientation(movable_handle, posObj=orig_movable_pos[0], ornObj=orig_movable_pos[1])

        return valid_so_far

    return unsupcol_check, static_obj_handles, movable_handles

# OPT: can reduce depth parameter to Poisson.
# OPT: one foreground mesh and one b
def create_mesh(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=5)

    # Helps to crop Poission surface extrapolations.
    # vertices_to_remove = densities < np.quantile(densities, 0.1)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Smoothen mesh.
    # mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)

    # Remove disconnected components.
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    largest_cluster_id = np.argmax(cluster_n_triangles)
    mesh.remove_triangles_by_mask(np.logical_not(triangle_clusters == largest_cluster_id))

    return mesh