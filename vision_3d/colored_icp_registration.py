import open3d as o3d
import numpy as np
import copy
import pcd_o3d
import argparse
from cfg import Config


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target])


if __name__ == '__main__':
    # set configs
    parser = argparse.ArgumentParser(description="Point Cloud Generator")
    parser.add_argument('--config', default='./configs/RealSense/config_rs_d435i.json', type=str)
    args = parser.parse_args()

    config_file = args.config
    cfg = Config(config_file)

    # From camera trajectory to fusing point cloud
    input_path = "/home/ryf/dataset/real_world/rgbd_test/rgbd_21_Mar_2023_14_16.395880/"
    color_path = input_path + "color/"
    depth_path = input_path + "depth/"
    pcd_path = input_path + "pcds/"
    pcd_out_path = input_path + "pcds_extrinsic/"
    extrinsic_file = input_path + "poses.txt"

    color_imgs, depth_imgs, ext_mats = pcd_o3d.load_rgbd_traj(color_path,
                                                              depth_path,
                                                              extrinsic_file)

    # set camera
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width=cfg.W,
                                                      height=cfg.H,
                                                      fx=cfg.fx,
                                                      fy=cfg.fy,
                                                      cx=cfg.cx,
                                                      cy=cfg.cy)

    source_color = color_imgs[51]
    source_depth = depth_imgs[51]
    target_color = color_imgs[52]
    target_depth = depth_imgs[52]
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth, convert_rgb_to_intensity=False)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        target_color, target_depth, convert_rgb_to_intensity=False)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        target_rgbd_image, intrinsic_o3d)

    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    option.depth_max = 3
    option.depth_diff_max = 0.01
    print(option)

    [success_color_term, trans_color_term, info] = \
        o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image,
        intrinsic_o3d, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    [success_hybrid_term, trans_hybrid_term, info] = \
        o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image,
        intrinsic_o3d, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

    if success_color_term:
        print("Using RGB-D Odometry")
        print(trans_color_term)
        source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, intrinsic_o3d)
        source_pcd_color_term.transform(trans_color_term)
        o3d.visualization.draw([target_pcd, source_pcd_color_term])

    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, intrinsic_o3d)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        o3d.visualization.draw([target_pcd, source_pcd_hybrid_term])

    print(ext_mats[51] * np.linalg.inv(ext_mats[52]))
    print(np.linalg.inv(ext_mats[51]) * ext_mats[52])
    # param = o3d.io.read_pinhole_camera_parameters("pcd/view_points/viewpoint_test.json")
    # param = o3d.camera.PinholeCameraParameters()
    # vis = o3d.visualization.Visualizer()
    # # vis.create_window()
    # vis.create_window(window_name="Point Cloud vis",
    #                   width=cfg.W,
    #                   height=cfg.H)
    #
    # ctr = vis.get_view_control()
    # p = ctr.convert_to_pinhole_camera_parameters()
    # print(p.intrinsic.intrinsic_matrix)