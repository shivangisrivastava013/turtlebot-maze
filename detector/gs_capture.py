#!/usr/bin/env python3
"""
Zenoh-based Gaussian Splat capture script for the TurtleBot/Gazebo house world.

Subscribes to:
  - camera/color/image_raw
  - camera/depth/image_rect_raw
  - odom

Writes a Nerfstudio-style RGB-D capture:
  output/
  ├── images/
  ├── depth/
  └── transforms.json

This script follows the same Zenoh + pycdr2 pattern as detector/object_detector.py.
"""

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import zenoh
from pycdr2 import IdlStruct
from pycdr2.types import uint8, uint32, int32, float64


# ----------------------------
# ROS CDR message definitions
# ----------------------------

@dataclass
class TimeMsg(IdlStruct, typename="builtin_interfaces/msg/Time"):
    sec: int32
    nanosec: uint32


@dataclass
class Header(IdlStruct, typename="std_msgs/msg/Header"):
    stamp: TimeMsg
    frame_id: str


@dataclass
class ImageMsg(IdlStruct, typename="sensor_msgs/msg/Image"):
    header: Header
    height: uint32
    width: uint32
    encoding: str
    is_bigendian: uint8
    step: uint32
    data: List[uint8]


@dataclass
class Point(IdlStruct, typename="geometry_msgs/msg/Point"):
    x: float64
    y: float64
    z: float64


@dataclass
class Quaternion(IdlStruct, typename="geometry_msgs/msg/Quaternion"):
    x: float64
    y: float64
    z: float64
    w: float64


@dataclass
class Pose(IdlStruct, typename="geometry_msgs/msg/Pose"):
    position: Point
    orientation: Quaternion


@dataclass
class PoseWithCovariance(IdlStruct, typename="geometry_msgs/msg/PoseWithCovariance"):
    pose: Pose
    covariance: List[float64]


@dataclass
class Vector3(IdlStruct, typename="geometry_msgs/msg/Vector3"):
    x: float64
    y: float64
    z: float64


@dataclass
class Twist(IdlStruct, typename="geometry_msgs/msg/Twist"):
    linear: Vector3
    angular: Vector3


@dataclass
class TwistWithCovariance(IdlStruct, typename="geometry_msgs/msg/TwistWithCovariance"):
    twist: Twist
    covariance: List[float64]


@dataclass
class OdometryMsg(IdlStruct, typename="nav_msgs/msg/Odometry"):
    header: Header
    child_frame_id: str
    pose: PoseWithCovariance
    twist: TwistWithCovariance


# ----------------------------
# Geometry helpers
# ----------------------------

def ros_time_to_sec(stamp: TimeMsg) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_to_rotmat(q: Quaternion) -> np.ndarray:
    """Convert geometry_msgs Quaternion to 3x3 rotation matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = x / n, y / n, z / n, w / n

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def odom_to_matrix(odom: OdometryMsg) -> np.ndarray:
    """T_odom_baselink from nav_msgs/Odometry."""
    T = np.eye(4, dtype=np.float64)
    p = odom.pose.pose.position
    q = odom.pose.pose.orientation
    T[:3, :3] = quat_to_rotmat(q)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def make_static_baselink_camera() -> np.ndarray:
    """
    Static TurtleBot base_link -> camera transform.

    Assignment spec offset:
      translation [0.064, -0.065, 0.094], no rotation
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.064, -0.065, 0.094]
    return T


def make_ros_optical_to_nerfstudio() -> np.ndarray:
    """
    Optical camera convention -> Nerfstudio/OpenGL camera convention.

    Assignment spec: 180-degree rotation around x-axis, flipping y and z.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    return T


def odom_to_nerfstudio_camera_matrix(odom: OdometryMsg) -> np.ndarray:
    """
    T_nerfstudio = T_odom_baselink @ T_baselink_camera @ T_ros_optical_to_nerfstudio
    """
    return odom_to_matrix(odom) @ make_static_baselink_camera() @ make_ros_optical_to_nerfstudio()


def yaw_from_odom(odom: OdometryMsg) -> float:
    """Extract planar yaw from odom quaternion."""
    q = odom.pose.pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


# ----------------------------
# Image conversion helpers
# ----------------------------

def image_msg_to_rgb(img_msg: ImageMsg) -> Optional[np.ndarray]:
    """Convert sensor_msgs/Image to RGB uint8 array."""
    enc = img_msg.encoding.lower()
    raw = np.frombuffer(bytes(img_msg.data), dtype=np.uint8)

    if enc == "rgb8":
        return raw.reshape(img_msg.height, img_msg.width, 3)

    if enc == "bgr8":
        bgr = raw.reshape(img_msg.height, img_msg.width, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if enc == "rgba8":
        rgba = raw.reshape(img_msg.height, img_msg.width, 4)
        return rgba[:, :, :3]

    if enc == "bgra8":
        bgra = raw.reshape(img_msg.height, img_msg.width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

    print(f"[WARN] Unsupported RGB encoding: {img_msg.encoding}", flush=True)
    return None


def image_msg_to_depth_mm(img_msg: ImageMsg) -> Optional[np.ndarray]:
    """Convert depth image to uint16 millimeters."""
    enc = img_msg.encoding.lower()
    data = bytes(img_msg.data)

    if enc in ["32fc1", "float32"]:
        depth_m = np.frombuffer(data, dtype=np.float32).reshape(img_msg.height, img_msg.width)
        depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
        return depth_mm

    if enc in ["16uc1", "mono16"]:
        return np.frombuffer(data, dtype=np.uint16).reshape(img_msg.height, img_msg.width)

    print(f"[WARN] Unsupported depth encoding: {img_msg.encoding}", flush=True)
    return None


# ----------------------------
# Capture state
# ----------------------------

class GSCapture:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output)
        self.image_dir = self.output_dir / "images"
        self.depth_dir = self.output_dir / "depth"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.latest_depth_msg: Optional[ImageMsg] = None
        self.latest_depth_wall_time: Optional[float] = None

        self.latest_odom_msg: Optional[OdometryMsg] = None
        self.latest_odom_wall_time: Optional[float] = None
        self.last_odom_warning_time = 0.0

        self.last_keyframe_position: Optional[np.ndarray] = None
        self.last_keyframe_yaw: Optional[float] = None

        self.frames = []
        self.frame_count = 0
        self.start_time = time.time()

        self.distance_thresh = args.distance_thresh
        self.angle_thresh_rad = math.radians(args.angle_thresh_deg)

    def should_capture(self, odom: OdometryMsg) -> bool:
        p = odom.pose.pose.position
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        yaw = yaw_from_odom(odom)

        if self.last_keyframe_position is None:
            return True

        dist = np.linalg.norm(pos[:2] - self.last_keyframe_position[:2])
        dyaw = angle_diff(yaw, self.last_keyframe_yaw)

        return dist >= self.distance_thresh or dyaw >= self.angle_thresh_rad

    def update_keyframe_pose(self, odom: OdometryMsg):
        p = odom.pose.pose.position
        self.last_keyframe_position = np.array([p.x, p.y, p.z], dtype=np.float64)
        self.last_keyframe_yaw = yaw_from_odom(odom)

    def on_depth(self, sample):
        try:
            self.latest_depth_msg = ImageMsg.deserialize(bytes(sample.payload))
            self.latest_depth_wall_time = time.time()
        except Exception as e:
            print(f"[WARN] Depth deserialize error: {e}", flush=True)

    def on_odom(self, sample):
        try:
            self.latest_odom_msg = OdometryMsg.deserialize(bytes(sample.payload))
            self.latest_odom_wall_time = time.time()
        except Exception as e:
            print(f"[WARN] Odom deserialize error: {e}", flush=True)

    def on_rgb(self, sample):
        now = time.time()

        try:
            rgb_msg = ImageMsg.deserialize(bytes(sample.payload))
        except Exception as e:
            print(f"[WARN] RGB deserialize error: {e}", flush=True)
            return

        if self.latest_depth_msg is None:
            return

        if self.latest_odom_msg is None:
            if now - self.last_odom_warning_time > 10.0:
                print("[WARN] No odom received yet; capture paused.", flush=True)
                self.last_odom_warning_time = now
            return

        if self.latest_depth_wall_time is None or now - self.latest_depth_wall_time > self.args.max_staleness:
            print("[WARN] Dropping RGB: depth is stale.", flush=True)
            return

        if self.latest_odom_wall_time is None or now - self.latest_odom_wall_time > self.args.max_staleness:
            print("[WARN] Dropping RGB: odom is stale.", flush=True)
            return

        odom = self.latest_odom_msg

        if not self.should_capture(odom):
            return

        rgb = image_msg_to_rgb(rgb_msg)
        depth_mm = image_msg_to_depth_mm(self.latest_depth_msg)

        if rgb is None or depth_mm is None:
            return

        self.frame_count += 1
        frame_id = f"{self.frame_count:05d}"

        rgb_path = self.image_dir / f"{frame_id}.png"
        depth_path = self.depth_dir / f"{frame_id}.png"

        # cv2 writes BGR, so convert RGB -> BGR.
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(depth_path), depth_mm)

        T = odom_to_nerfstudio_camera_matrix(odom)

        frame = {
            "file_path": f"images/{frame_id}.png",
            "depth_file_path": f"depth/{frame_id}.png",
            "transform_matrix": T.tolist(),
            "timestamp": ros_time_to_sec(rgb_msg.header.stamp),
            "frame_id": rgb_msg.header.frame_id,
        }

        self.frames.append(frame)
        self.update_keyframe_pose(odom)
        self.write_transforms()

        print(
            f"[CAPTURE] #{self.frame_count} rgb={rgb.shape[1]}x{rgb.shape[0]} "
            f"depth={depth_mm.shape[1]}x{depth_mm.shape[0]} output={rgb_path}",
            flush=True,
        )

    def write_transforms(self):
        transforms = {
            "camera_model": "OPENCV",
            "fl_x": self.args.fx,
            "fl_y": self.args.fy,
            "cx": self.args.cx,
            "cy": self.args.cy,
            "w": self.args.width,
            "h": self.args.height,
            "frames": self.frames,
        }

        out = self.output_dir / "transforms.json"
        tmp = self.output_dir / "transforms.json.tmp"
        tmp.write_text(json.dumps(transforms, indent=2))
        tmp.replace(out)

    def write_summary(self):
        elapsed = time.time() - self.start_time
        summary = {
            "frames_captured": self.frame_count,
            "elapsed_seconds": elapsed,
            "distance_threshold_m": self.distance_thresh,
            "angle_threshold_deg": self.args.angle_thresh_deg,
            "output_dir": str(self.output_dir),
        }
        summary_path = self.output_dir / "capture_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Zenoh RGB-D Gaussian Splat Capture")
    parser.add_argument("--connect", type=str, default="tcp/localhost:7447")
    parser.add_argument("--rgb-key", type=str, default="camera/color/image_raw")
    parser.add_argument("--depth-key", type=str, default="camera/depth/image_rect_raw")
    parser.add_argument("--odom-key", type=str, default="odom")
    parser.add_argument("--output", type=str, default="/data/captures/house_run")
    parser.add_argument("--distance-thresh", type=float, default=0.3)
    parser.add_argument("--angle-thresh-deg", type=float, default=10.0)
    parser.add_argument("--max-staleness", type=float, default=0.2)

    # Simulated D435i intrinsics from assignment.
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fx", type=float, default=277.13)
    parser.add_argument("--fy", type=float, default=277.13)
    parser.add_argument("--cx", type=float, default=160.0)
    parser.add_argument("--cy", type=float, default=120.0)

    return parser.parse_args()


def main():
    args = parse_args()

    print("Gaussian Splat capture starting.", flush=True)
    print(f"RGB key:   {args.rgb_key}", flush=True)
    print(f"Depth key: {args.depth_key}", flush=True)
    print(f"Odom key:  {args.odom_key}", flush=True)
    print(f"Output:    {args.output}", flush=True)

    capture = GSCapture(args)

    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps([args.connect]))

    session = zenoh.open(conf)

    sub_depth = session.declare_subscriber(args.depth_key, capture.on_depth)
    sub_odom = session.declare_subscriber(args.odom_key, capture.on_odom)
    sub_rgb = session.declare_subscriber(args.rgb_key, capture.on_rgb)

    def shutdown(sig, frame):
        print("\nStopping capture.", flush=True)
        capture.write_transforms()
        capture.write_summary()
        sub_rgb.undeclare()
        sub_depth.undeclare()
        sub_odom.undeclare()
        session.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Capture running. Press Ctrl+C to stop.", flush=True)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()