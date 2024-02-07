import re
import os
import json
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert accio camera format poses.txt to instant-ngp format.")

    parser.add_argument("--folder_path", default="../dataset/posed_rgbds", help="Input path.")
    args = parser.parse_args()
    return args


def closest_point_2_lines(oa, da, ob, db):
    # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def raw_poses_convert(cfg, OUT_PATH):
    out = {
        "fl_x": cfg.fx,
        "fl_y": cfg.fy,
        "k1": cfg.k1,
        "k2": cfg.k2,
        "k3": cfg.k3,
        "k4": cfg.k4,
        "p1": cfg.p1,
        "p2": cfg.p2,
        "is_fisheye": cfg.is_fisheye,
        "cx": cfg.cx,
        "cy": cfg.cy,
        "w": cfg.W,
        "h": cfg.H,
        "aabb_scale": 2,
        "scale": cfg.scale,
        "offset": cfg.offset,  # BRG (xyz)
        "frames": []
    }

    if cfg.camera_angle_x is not None:
        out["camera_angle_x"] = cfg.camera_angle_x
        out["camera_angle_y"] = cfg.camera_angle_y
        out["camera_angle_x"] = cfg.camera_angle_x
        out["camera_angle_y"] = cfg.camera_angle_y

    transforms_file = os.path.join(cfg.data_dir, "poses.txt")
    traj_ext = np.loadtxt(transforms_file, delimiter=" ").reshape([-1, 4, 4])
    print(len(traj_ext))

    up = np.zeros(3)
    for i in range(len(traj_ext)):
        name = "./images/rgb_%04d.png" % i
        # name = "./images/frame%06d.png" % i

        c2w = traj_ext[i]
        c2w[:3, 1] *= -1  # flip the x and z axis
        c2w[:3, 2] *= -1
        up += c2w[:3, 1]

        frame = {"file_path": name, "transform_matrix": c2w.tolist()}
        out["frames"].append(frame)

    # # Reorient the scene to be easier to work with
    # nframes = len(out["frames"])
    # up = up / np.linalg.norm(up)
    # print("up vector was", up)
    # R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    # R = np.pad(R, [0, 1])
    # R[-1, -1] = 1
    #
    # for f in out["frames"]:
    #     f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis
    #
    # # find a central point they are all looking at
    # print("computing center of attention...")
    # totw = 0.0
    # totp = np.array([0.0, 0.0, 0.0])
    # for f in out["frames"]:
    #     mf = f["transform_matrix"][0:3, :]
    #     for g in out["frames"]:
    #         mg = g["transform_matrix"][0:3, :]
    #         p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
    #         if w > 0.00001:
    #             totp += p * w
    #             totw += w
    # if totw > 0.0:
    #     totp /= totw
    # print(totp)  # the cameras are looking at totp
    # for f in out["frames"]:
    #     f["transform_matrix"][0:3, 3] -= totp
    #
    # avglen = 0.
    # for f in out["frames"]:
    #     avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    # avglen /= nframes
    # print("avg camera distance from origin", avglen)
    # for f in out["frames"]:
    #     f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"
    #
    # for f in out["frames"]:
    #     f["transform_matrix"] = f["transform_matrix"].tolist()

    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)


def converter(T_accio_list):
    # flip the y and z axis
    T_ngp_list = T_accio_list.copy()
    for T_ngp in T_ngp_list:
        T_ngp[:3, 1] *= -1
        T_ngp[:3, 2] *= -1
    return T_ngp_list


if __name__ == '__main__':
    args = parse_args()
    transforms_file = os.path.join(args.folder_path, "poses.txt")
    OUT_PATH = os.path.join(args.folder_path, "transforms.json")
    raw_poses_convert(transforms_file, OUT_PATH)


