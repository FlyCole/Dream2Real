import open3d as o3d
import numpy as np
import pcd_o3d
import argparse
from cfg import Config


def gen_pcd_with_extrinsics(color_file, depth_file, intrinsic, extrinsic=None):
    color_raw = o3d.io.read_image(color_file)
    depth_raw = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                    convert_rgb_to_intensity=False)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic)

    return point_cloud


def load_point_clouds(pcds_path, voxel_size=0.0):
    pcds = []
    for i in range(63):
        pcd_file = pcds_path + "pcd%06d.pcd" % i
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals()
        pcds.append(pcd_down)

    return pcds


voxel_size = 0.001
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


if __name__ == '__main__':
    # set configs
    parser = argparse.ArgumentParser(description="Point Cloud Generator")
    parser.add_argument('--config', default='./configs/RealSense/config_rs_d435i.json', type=str)
    args = parser.parse_args()

    config_file = args.config
    cfg = Config(config_file)

    # set camera
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width=cfg.W,
                                                      height=cfg.H,
                                                      fx=cfg.fx,
                                                      fy=cfg.fy,
                                                      cx=cfg.cx,
                                                      cy=cfg.cy)

    input_path = "/home/ryf/dataset/real_world/rgbd_test/rgbd_21_Mar_2023_14_16.395880/"
    color_path = input_path + "color/"
    depth_path = input_path + "depth/"
    pcd_path = input_path + "pcds/"
    pcd_out_path = input_path + "pcds_extrinsic/"
    extrinsic_file = input_path + "poses.txt"

    # From camera trajectory to fusing point cloud
    # ext_mats = []
    # with open(extrinsic_file, 'r') as file:
    #     i = 0
    #     mat = np.identity(4)
    #
    #     for line in file:
    #         data_line = line.strip("\n").split()
    #         if i == 4:
    #             ext_mats.append(np.linalg.inv(mat))
    #             mat = np.identity(4)
    #             i = 0
    #         mat[i, :] = data_line[:]
    #         i += 1
    #
    # pcds = []
    # for i in range(62):
    #     print(ext_mats[i])
    #     color_file = color_path + "frame%06d.png" % i
    #     depth_file = depth_path + "depth%06d.png" % i
    #     pcd = gen_pcd_with_extrinsics(color_file, depth_file, intrinsic_o3d, ext_mats[i])
    #     pcds.append(pcd)
    #
    # o3d.visualization.draw_geometries(pcds,
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    # generate point clouds
    # for i in range(63):
    #     color_file = color_path + "frame%06d.png" % i
    #     depth_file = depth_path + "depth%06d.png" % i
    #     pcd_out_file = pcd_out_path + "pcd%06d.pcd" % i
    #
    #     pcd = gen_pcd_with_extrinsics(color_file, depth_file, intrinsic_o3d)
    #     o3d.io.write_point_cloud(pcd_out_file, pcd)


    voxel_size = 0.001
    pcds_down = load_point_clouds(pcd_path, voxel_size)
    o3d.visualization.draw_geometries(pcds_down,
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3d.visualization.draw_geometries(pcds_down,
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])