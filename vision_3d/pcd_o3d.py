import os

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import argparse
from cfg import Config
import open3d.visualization.rendering as rendering


def load_rgbd_traj(color_path, depth_path, extrinsic_file):
    color_imgs = []
    depth_imgs = []
    # ext_mats = []
    # with open(extrinsic_file, 'r') as file:
    #     i = 0
    #     mat = np.identity(4)
    #
    #     for line in file:
    #         # save extrinsic matrix
    #         data_line = line.strip("\n").split()
    #         if i == 4:
    #             ext_mats.append(np.linalg.inv(mat))
    #             mat = np.identity(4)
    #             i = 0
    #         mat[i, :] = data_line[:]
    #         i += 1

    ext_mats = np.loadtxt(extrinsic_file, delimiter=" ").reshape([-1, 4, 4])
    print(ext_mats)

    # save RGB-D image
    for index in range(len(ext_mats)):
        # color_file = os.path.join(color_path, "frame%06d.png" % index)
        # depth_file = os.path.join(depth_path, "depth%06d.png" % index)
        color_file = os.path.join(color_path, "rgb_%04d.png" % index)
        depth_file = os.path.join(depth_path, "depth_%04d.png" % index)
        color_raw = o3d.io.read_image(color_file)
        depth_raw = o3d.io.read_image(depth_file)
        color_imgs.append(color_raw)
        depth_imgs.append(depth_raw)

    return color_imgs, depth_imgs, ext_mats


def rgbd_to_o3d_pcd(color_path, depth_path, intrinsic, data_format=None):
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)

    if data_format is None:
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                        convert_rgb_to_intensity=False)

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    elif data_format == "TUM":
        rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw,
                                                                   convert_rgb_to_intensity=False)
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    else:
        print("Data format {} is not implemented".format(data_format))
        return

    # plt.subplot(1, 2, 1)
    # plt.title('Test color image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Test depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()

    # Flip it, otherwise the pointcloud will be upside down
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return point_cloud


def posed_rgbd_to_o3d_pcd(color_img, depth_img, intrinsic, extrinsic):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img,
                                                                    convert_rgb_to_intensity=False)

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic)
    # Flip it, otherwise the pointcloud will be upside down
    # point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return point_cloud


def mask_rgbd_to_o3d_pcd(color_path, depth_path, mask_path, intrinsic, mask_type=0, data_format=None):
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)
    mask = np.load(mask_path)

    # mask out
    if mask_type == 0:
        # keep the point mask[i] == 0
        np.asarray(color_raw)[mask[0]] = 0
        np.asarray(depth_raw)[mask[0]] = 0
    elif mask_type == 1:
        # keep the point mask[i] == 1
        np.asarray(color_raw)[~mask[0]] = 0
        np.asarray(depth_raw)[~mask[0]] = 0

    if data_format is None:
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                        convert_rgb_to_intensity=False)

        mask_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    elif data_format == "TUM":
        rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw,
                                                                   convert_rgb_to_intensity=False)
        mask_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    else:
        print("Data format {} is not implemented".format(data_format))
        return

    # Flip it, otherwise the pointcloud will be upside down
    mask_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return mask_pcd


def save_view_point(pcd, filename, w=800, h=600):
    """
    Save the current view point.
    :param pcd: current point cloud
    :param filename: output file name
    :param w: image width
    :param h: image height
    :return:
    """
    vis_temp = o3d.visualization.Visualizer()
    vis_temp.create_window(window_name='pcd', width=w, height=h)
    vis_temp.add_geometry(pcd)
    vis_temp.run()   # user changes the view and press "q" to terminate
    param = vis_temp.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis_temp.destroy_window()


def load_view_point(pcd, filename, w=800, h=600):
    """
    Load the saved view point.
    :param pcd: current point cloud
    :param filename: input file name
    :param w: image width
    :param h: image height
    :return:
    """
    vis_temp = o3d.visualization.Visualizer()
    vis_temp.create_window(window_name='pcd', width=w, height=h)
    ctr = vis_temp.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis_temp.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis_temp.run()
    vis_temp.destroy_window()


def pcd_to_mesh(pcd):
    """
    Transform a point cloud to mesh
    :param pcd: given point cloud
    :return: output mesh
    """
    pcd.estimate_normals()

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2]))
    # print(mesh.get_surface_area())

    o3d.visualization.draw_geometries([mesh], window_name='Open3D downSample', width=800, height=600, left=50,
                                      top=50, point_show_normal=True, mesh_show_wireframe=True,
                                      mesh_show_back_face=True, )

    return mesh


def render_from_viewpoint(inputs, output_path, intrinsic, extrinsic):
    """
    Render a pcd / mesh from different view point with a mask for invalid depth
    :param inputs: given point cloud or mesh
    :param intrinsic: camera intrinsic in o3d format
    :param extrinsic: camera extrinsic in numpy format
    :return:
        1. color: rendered color image
        2. invalid_depth_mask: invalid mask of depth
    """
    render = rendering.OffscreenRenderer(intrinsic.width, intrinsic.height)
    render.scene.add_geometry("input", inputs, rendering.MaterialRecord())
    # render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])
    # render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0], 75000)
    # render.scene.scene.enable_sun_light(False)
    # render.scene.show_axes(True)

    render.setup_camera(intrinsic, extrinsic)
    color = render.render_to_image()
    depth = render.render_to_depth_image()

    invalid_depth_mask = np.where(np.asarray(depth) == 1, 0, 1).astype(np.uint8)

    color_path = output_path + "rendered_color.png"
    print("Saving image at {}".format(color_path))
    o3d.io.write_image(color_path, color)

    invalid_path = output_path + "invalid_depth_map.png"
    plt.imsave(invalid_path, invalid_depth_mask)

    return color, invalid_depth_mask


def pcd_trans_rot(pcd, translation, rotation=None):
    pcd.translate(translation, relative=True)
    pcd.rotate(rotation, center=(0, 0, 0))


def pcd_distance(source_pcd, target_pcd):
    dists = source_pcd.compute_point_cloud_distance(target_pcd)
    dists = np.asarray(dists)
    print(dists.shape)
    print(dists)
    min_dist = np.min(dists)
    ind = np.where(dists < 0.03)[0]
    print(ind.shape)
    print(min_dist)


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

    # color_path = "./test_data/rgb/1305032353.993165.png"
    # depth_path = "./test_data/depth/1305032354.009456.png"
    # mask_path = "./test_data/mask/TUM.npy"
    color_path = "./test_data/rgb/frame000052.png"
    depth_path = "./test_data/depth/depth000052.png"
    mask_path = "./test_data/mask/panda.npy"

    pcd = rgbd_to_o3d_pcd(color_path, depth_path, intrinsic_o3d)
    # pcd = mask_rgbd_to_o3d_pcd(color_path, depth_path, mask_path, intrinsic_o3d, 1)
    # pcd = rgbd_to_o3d_pcd(color_path, depth_path, intrinsic_o3d, data_format="TUM")
    # pcd = mask_rgbd_to_o3d_pcd(color_path, depth_path, mask_path, intrinsic_o3d, data_format="TUM")

    # set visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud vis",
                      width=cfg.W,
                      height=cfg.H)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    # visualize from different view point
    # save_view_point(pcd, "viewpoint_11.json", w=cfg.W, h=cfg.H)
    load_view_point(pcd, "pcd/view_points/viewpoint_top.json", w=cfg.W, h=cfg.H)
    # o3d.io.write_point_cloud("./pcd/pcd_table.pcd", pcd)

    # change pcd to mesh and render from mesh
    # pcd_mesh = copy.deepcopy(pcd)
    # mesh = pcd_to_mesh(pcd_mesh)

    # render from pcd / mesh from different view point
    extrinsic = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # render_from_viewpoint(pcd, "test_data/render/01.png", intrinsic_o3d, extrinsic)
