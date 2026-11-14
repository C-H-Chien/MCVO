#!/usr/bin/env python3
"""Convert a Waymo extracted sequence to an MCVO-compatible rosbag.

Expects the layout produced by waymo_data_parser.py:

  <DATASET>/data/<SEQUENCE>/
    FRONT/*.jpg
    FRONT_LEFT/*.jpg
    FRONT_RIGHT/*.jpg
    SIDE_LEFT/*.jpg
    SIDE_RIGHT/*.jpg
    timestamps.txt          # frame_idx  timestamp_micros  timestamp_seconds
    camera_calibration.json
    gt_poses.txt

Writes synchronized sensor_msgs/Image (bgr8) topics:
  /camera_00  FRONT
  /camera_01  FRONT_LEFT
  /camera_02  FRONT_RIGHT
  /camera_03  SIDE_LEFT
  /camera_04  SIDE_RIGHT

Requires ROS Noetic Python libs (rosbag, rospy, sensor_msgs).
Run inside the MCVO Apptainer image after ROS is sourced, e.g.:

  apptainer exec \\
    -B /gpfs/data/bkimia/Datasets/Waymo:/gpfs/data/bkimia/Datasets/Waymo \\
    -B /gpfs/data/bkimia/cchien3/MCVO-forked/MCVO:/gpfs/data/bkimia/cchien3/MCVO-forked/MCVO \\
    /gpfs/data/bkimia/cchien3/mcvo.simg bash -c \\
    'source /opt/ros/noetic/setup.bash; \\
     python3 /gpfs/data/bkimia/cchien3/MCVO-forked/MCVO/rosbag_converters/waymo_to_mcvo_bag.py \\
       --sequence 100613'
"""
from __future__ import print_function

import argparse
import os
import sys


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
            "Example:\n"
            "  apptainer exec /gpfs/data/bkimia/cchien3/mcvo.simg bash -c \\\n"
            "    'source /opt/ros/noetic/setup.bash; python3 waymo_to_mcvo_bag.py'\n".format(e)
        )
        sys.exit(1)


# Waymo camera folder -> (MCVO camera id, ROS topic)
CAMERAS = [
    ("FRONT",       "00", "/camera_00"),
    ("FRONT_LEFT",  "01", "/camera_01"),
    ("FRONT_RIGHT", "02", "/camera_02"),
    ("SIDE_LEFT",   "03", "/camera_03"),
    ("SIDE_RIGHT",  "04", "/camera_04"),
]

DEFAULT_DATASET = "/gpfs/data/bkimia/Datasets/Waymo"
DEFAULT_OUT_DIR = "/gpfs/data/bkimia/cchien3/MCVO-forked/MCVO/rosbags"


def read_timestamps(seq_dir):
    """Parse timestamps.txt -> list of (micros_str, seconds_float)."""
    path = os.path.join(seq_dir, "timestamps.txt")
    stamps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError("bad timestamps.txt line: {!r}".format(line))
            micros = parts[1]
            seconds = float(parts[2])
            stamps.append((micros, seconds))
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="Waymo root containing data/<SEQUENCE>/ (default: %(default)s)")
    p.add_argument("--sequence", default="100613",
                   help="Sequence id under data/ (default: %(default)s)")
    p.add_argument("--out-bag", default=None,
                   help="Output .bag path (default: <out-dir>/Waymo_<sequence>.bag)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="Default output directory when --out-bag is omitted")
    p.add_argument("--start-frame", type=int, default=0,
                   help="First frame index inclusive (default: 0)")
    p.add_argument("--end-frame", type=int, default=-1,
                   help="Last frame index inclusive (-1 = last frame)")
    p.add_argument("--relative-time", action="store_true",
                   help="Write stamps relative to first frame (+1ms offset) "
                        "instead of absolute Waymo epoch times")
    return p.parse_args()


def main():
    args = parse_args()
    cv2, rosbag, rospy, Image = _require_ros_imports()

    seq_dir = os.path.join(args.dataset, "data", args.sequence)
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError("sequence directory not found: {}".format(seq_dir))

    stamps = read_timestamps(seq_dir)
    if not stamps:
        raise RuntimeError("no timestamps in {}".format(
            os.path.join(seq_dir, "timestamps.txt")))

    start = args.start_frame
    end = args.end_frame if args.end_frame >= 0 else (len(stamps) - 1)
    if start < 0 or end >= len(stamps) or start > end:
        raise IndexError(
            "invalid frame range [{}, {}] for {} timestamps".format(
                start, end, len(stamps)))

    out_bag = args.out_bag or os.path.join(
        args.out_dir, "Waymo_{}.bag".format(args.sequence))
    os.makedirs(os.path.dirname(out_bag) or ".", exist_ok=True)

    t0 = stamps[start][1]
    TIME_OFFSET_SEC = 1e-3  # avoid invalid zero Time when using relative stamps

    n_frames = end - start + 1
    print("Sequence : {}".format(args.sequence))
    print("Frames   : {} [{}..{}]".format(n_frames, start, end))
    print("Cameras  : {}".format(", ".join(c[0] for c in CAMERAS)))
    print("Topics   : {}".format(", ".join(c[2] for c in CAMERAS)))
    print("Time mode: {}".format("relative" if args.relative_time else "absolute epoch"))
    print("Output   : {}".format(out_bag))
    print("python   : {}".format(sys.executable))

    with rosbag.Bag(out_bag, "w") as bag:
        for i in range(start, end + 1):
            micros, seconds = stamps[i]
            if args.relative_time:
                rel = (seconds - t0) + TIME_OFFSET_SEC
                t = rospy.Time.from_sec(rel)
            else:
                t = rospy.Time.from_sec(seconds)

            if t.secs == 0 and t.nsecs == 0:
                raise ValueError("refusing to write zero Time at frame {}".format(i))

            for cam_name, cam_id, topic in CAMERAS:
                img_path = os.path.join(seq_dir, cam_name, "{}.jpg".format(micros))
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(img_path)
                msg = bgr_to_imgmsg(img, t, "camera_{}".format(cam_id), Image)
                bag.write(topic, msg, t)

            if (i - start) % 20 == 0 or i == end:
                print("frame {:4d}/{:4d}  t={:.6f}s".format(
                    i, end, t.to_sec()))

    print("Wrote", out_bag)


if __name__ == "__main__":
    main()
