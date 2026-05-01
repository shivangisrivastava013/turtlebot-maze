#!/usr/bin/env python3
"""
Convert gs_capture.py Nerfstudio-style RGB-D capture into a COLMAP text model.

Input:
  data/captures/house_run/
  ├── images/
  ├── depth/
  └── transforms.json

Output:
  data/captures/house_run_colmap/
  ├── images/
  └── sparse/0/
      ├── cameras.txt
      ├── images.txt
      └── points3D.txt

Notes:
- gs_capture.py writes transform_matrix in Nerfstudio/OpenGL camera convention.
- COLMAP expects camera-to-world/world-to-camera in OpenCV camera convention.
- We undo the y/z flip before writing COLMAP poses.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to COLMAP qvec format: qw, qx, qy, qz.
    """
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ],
        dtype=np.float64,
    )
    K /= 3.0

    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]

    if q[0] < 0:
        q *= -1

    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input house_run folder")
    parser.add_argument("--output", required=True, help="Output COLMAP folder")
    parser.add_argument("--point-stride", type=int, default=20, help="Depth pixel stride for point cloud sampling")
    parser.add_argument("--max-points-per-frame", type=int, default=250, help="Maximum sampled depth points per frame")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    transforms_path = input_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(transforms_path)

    with open(transforms_path, "r") as f:
        meta = json.load(f)

    frames = meta["frames"]
    width = int(meta["w"])
    height = int(meta["h"])
    fx = float(meta["fl_x"])
    fy = float(meta["fl_y"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])

    image_out = output_dir / "images"
    sparse_out = output_dir / "sparse" / "0"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    image_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    # Copy RGB images into output/images with same names.
    for frame in frames:
        src = input_dir / frame["file_path"]
        dst = image_out / Path(frame["file_path"]).name
        shutil.copy(src, dst)

    # COLMAP camera model.
    cameras_txt = sparse_out / "cameras.txt"
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}\n")

    # Nerfstudio/OpenGL to COLMAP/OpenCV camera convention correction.
    flip = np.eye(4, dtype=np.float64)
    flip[:3, :3] = np.diag([1.0, -1.0, -1.0])

    images_txt = sparse_out / "images.txt"
    points_txt = sparse_out / "points3D.txt"

    points = []
    point_id = 1

    with open(images_txt, "w") as f_img:
        f_img.write("# Image list with two lines of data per image:\n")
        f_img.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f_img.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, frame in enumerate(frames, start=1):
            T_ns_c2w = np.array(frame["transform_matrix"], dtype=np.float64)

            # Convert Nerfstudio/OpenGL C2W back to COLMAP/OpenCV C2W.
            T_cv_c2w = T_ns_c2w @ flip

            # COLMAP stores world-to-camera.
            T_w2c = np.linalg.inv(T_cv_c2w)
            R_w2c = T_w2c[:3, :3]
            t_w2c = T_w2c[:3, 3]

            qvec = rotmat_to_qvec(R_w2c)
            image_name = Path(frame["file_path"]).name

            f_img.write(
                f"{idx} "
                f"{qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
                f"{t_w2c[0]:.12f} {t_w2c[1]:.12f} {t_w2c[2]:.12f} "
                f"1 {image_name}\n"
            )

            # No 2D tracks. This is acceptable for random-init training and still gives camera poses.
            f_img.write("\n")

            # Sample sparse 3D points from depth for initialization.
            rgb_path = input_dir / frame["file_path"]
            depth_path = input_dir / frame["depth_file_path"]

            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

            if rgb is None or depth is None:
                continue

            # RGB is BGR from OpenCV.
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            sampled_this_frame = 0

            for v in range(0, min(height, depth.shape[0]), args.point_stride):
                for u in range(0, min(width, depth.shape[1]), args.point_stride):
                    if sampled_this_frame >= args.max_points_per_frame:
                        break

                    z_mm = float(depth[v, u])
                    if z_mm <= 0 or not math.isfinite(z_mm):
                        continue

                    z = z_mm / 1000.0
                    if z < 0.1 or z > 10.0:
                        continue

                    x = (u - cx) / fx * z
                    y = (v - cy) / fy * z

                    p_cam = np.array([x, y, z, 1.0], dtype=np.float64)
                    p_world = T_cv_c2w @ p_cam

                    color = rgb[v, u]
                    points.append(
                        (
                            point_id,
                            p_world[0],
                            p_world[1],
                            p_world[2],
                            int(color[0]),
                            int(color[1]),
                            int(color[2]),
                        )
                    )
                    point_id += 1
                    sampled_this_frame += 1

                if sampled_this_frame >= args.max_points_per_frame:
                    break

    with open(points_txt, "w") as f_pts:
        f_pts.write("# 3D point list with one line of data per point:\n")
        f_pts.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f_pts.write(f"# Number of points: {len(points)}\n")

        for pid, x, y, z, r, g, b in points:
            # Empty track is intentional.
            f_pts.write(f"{pid} {x:.8f} {y:.8f} {z:.8f} {r} {g} {b} 1.0\n")

    print("Converted capture to COLMAP text format.")
    print("Input frames:", len(frames))
    print("Output:", output_dir)
    print("Images:", len(list(image_out.glob('*'))))
    print("Sparse:", sparse_out)
    print("3D points:", len(points))
    print("cameras.txt:", cameras_txt.exists())
    print("images.txt:", images_txt.exists())
    print("points3D.txt:", points_txt.exists())


if __name__ == "__main__":
    main()