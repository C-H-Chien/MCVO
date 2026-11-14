#!/usr/bin/env python3
"""Convert KITTI-360 sequence to MCVO-compatible rosbag.

Requires ROS Noetic Python libs (rosbag, rospy, sensor_msgs).
Run inside the MCVO Apptainer image after ROS is sourced, e.g.:

  ./run_mcvo.sh
  # inside Apptainer (do NOT use a host conda env):
  python3 kitti360_to_mcvo_bag.py

Writes topics: /camera_00, /camera_01, /camera_02, /camera_03
Timestamps use rospy.Time.from_sec() so bag indices are valid.
"""
from __future__ import print_function

import os
import sys
from datetime import datetime

def _require_ros_imports():
    try:
        import cv2
        import rosbag
        import rospy
        from sensor_msgs.msg import Image
        return cv2, rosbag, rospy, Image
    except ImportError as e:
        sys.stderr.write(
            "Missing ROS/OpenCV Python module: {}\n\n"
            "This script must run inside the MCVO Apptainer image with ROS sourced.\n"
            "On the GPU node:\n"
            "  conda deactivate   # if a conda env is active\n"
            "  ./run_mcvo.sh\n"
            "  python3 kitti360_to_mcvo_bag.py\n".format(e)
        )
        sys.exit(1)


DATASET = "/gpfs/data/bkimia/Datasets/KITTI-360"
SEQUENCE = "2013_05_28_drive_0000_sync"
OUT_BAG = "/gpfs/data/bkimia/cchien3/MCVO-forked/MCVO/rosbags/KITTI360_00_1.bag"
START_FRAME = 0
END_FRAME = 2277  # inclusive, per README example

CAMERAS = [
    ("00", "/camera_00", "data_rect"),  # perspective left
    ("01", "/camera_01", "data_rect"),  # perspective right
    ("02", "/camera_02", "data_rgb"),   # fisheye left
    ("03", "/camera_03", "data_rgb"),   # fisheye right
]


def read_timestamps(seq, cam="00"):
    """Parse KITTI-360 timestamps.txt lines like: 2013-05-28 08:46:02.904483072"""
    path = os.path.join(DATASET, "data_2d_raw", seq, "image_{}".format(cam), "timestamps.txt")
    stamps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            date_str, time_str = line.split()
            if "." in time_str:
                whole, frac = time_str.split(".", 1)
                time_str = "{}.{}".format(whole, frac[:6].ljust(6, "0"))
            dt = datetime.strptime("{} {}".format(date_str, time_str), "%Y-%m-%d %H:%M:%S.%f")
            stamps.append((dt - datetime(1970, 1, 1)).total_seconds())
    return stamps


def bgr_to_imgmsg(img, stamp, frame_id, Image):
    """Build sensor_msgs/Image without cv_bridge."""
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = img.tobytes()
    return msg


def main():
    cv2, rosbag, rospy, Image = _require_ros_imports()

    stamps = read_timestamps(SEQUENCE, "00")
    if END_FRAME >= len(stamps):
        raise IndexError(
            "END_FRAME={} but only {} timestamps in timestamps.txt".format(
                END_FRAME, len(stamps))
        )

    t0 = stamps[START_FRAME]
    os.makedirs(os.path.dirname(OUT_BAG), exist_ok=True)

    print("Writing {} frames [{}..{}] -> {}".format(
        END_FRAME - START_FRAME + 1, START_FRAME, END_FRAME, OUT_BAG))
    print("python:", sys.executable)

    # rospy.Time(0) / from_sec(0.0) is invalid in rosbag indices ("invalid time").
    # Shift relative stamps so the first frame is never exactly zero.
    TIME_OFFSET_SEC = 1e-3

    with rosbag.Bag(OUT_BAG, "w") as bag:
        for i in range(START_FRAME, END_FRAME + 1):
            rel = (stamps[i] - t0) + TIME_OFFSET_SEC
            t = rospy.Time.from_sec(rel)
            if t.secs == 0 and t.nsecs == 0:
                raise ValueError("refusing to write zero Time at frame {}".format(i))

            for cam_id, topic, subdir in CAMERAS:
                img_path = os.path.join(
                    DATASET, "data_2d_raw", SEQUENCE,
                    "image_{}".format(cam_id), subdir, "{:010d}.png".format(i),
                )
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(img_path)
                msg = bgr_to_imgmsg(img, t, "camera_{}".format(cam_id), Image)
                bag.write(topic, msg, t)

            if i % 100 == 0:
                print("frame {}  t={:.6f}s".format(i, t.to_sec()))

    print("Wrote", OUT_BAG)


if __name__ == "__main__":
    main()
