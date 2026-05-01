import math
import numpy as np

from gs_capture import (
    TimeMsg,
    Header,
    Point,
    Quaternion,
    Pose,
    PoseWithCovariance,
    Vector3,
    Twist,
    TwistWithCovariance,
    OdometryMsg,
    odom_to_nerfstudio_camera_matrix,
)


def make_odom(x=0.0, y=0.0, z=0.0, yaw=0.0):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    return OdometryMsg(
        header=Header(stamp=TimeMsg(sec=0, nanosec=0), frame_id="odom"),
        child_frame_id="base_link",
        pose=PoseWithCovariance(
            pose=Pose(
                position=Point(x=x, y=y, z=z),
                orientation=Quaternion(x=0.0, y=0.0, z=qz, w=qw),
            ),
            covariance=[0.0] * 36,
        ),
        twist=TwistWithCovariance(
            twist=Twist(
                linear=Vector3(x=0.0, y=0.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            covariance=[0.0] * 36,
        ),
    )


def test_identity_odom_static_camera_transform():
    odom = make_odom()
    T = odom_to_nerfstudio_camera_matrix(odom)

    expected = np.eye(4)
    expected[:3, :3] = np.diag([1.0, -1.0, -1.0])
    expected[:3, 3] = [0.064, -0.065, 0.094]

    np.testing.assert_allclose(T, expected, atol=1e-6)


def test_translated_odom_adds_base_translation():
    odom = make_odom(x=1.0, y=2.0, z=0.0)
    T = odom_to_nerfstudio_camera_matrix(odom)

    expected_translation = np.array([1.064, 1.935, 0.094])
    np.testing.assert_allclose(T[:3, 3], expected_translation, atol=1e-6)


def test_yaw_90_rotates_camera_offset():
    odom = make_odom(x=0.0, y=0.0, z=0.0, yaw=math.pi / 2.0)
    T = odom_to_nerfstudio_camera_matrix(odom)

    # base_link camera offset [0.064, -0.065, 0.094] rotated 90 degrees about z:
    # x' = -y = 0.065
    # y' =  x = 0.064
    expected_translation = np.array([0.065, 0.064, 0.094])
    np.testing.assert_allclose(T[:3, 3], expected_translation, atol=1e-6)