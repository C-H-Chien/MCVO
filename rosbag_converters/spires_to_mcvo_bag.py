#!/usr/bin/env python3
"""Convert an Oxford Spires ROS1 bag to an MCVO-compatible rosbag.

The Spires bags store JPEG-compressed color images
(sensor_msgs/CompressedImage) on:
    /alphasense_driver_ros/cam0/debayered/image/compressed  (front)
    /alphasense_driver_ros/cam1/debayered/image/compressed  (left)
    /alphasense_driver_ros/cam2/debayered/image/compressed  (right)

MCVO's frontend subscribes to raw sensor_msgs/Image, so this script decodes
each JPEG and rewrites it as a bgr8 Image on /camera_00 .. /camera_02.
Original header timestamps are kept (they are valid, nonzero epoch times).

The cameras are hardware-synchronized: shared frames have IDENTICAL header
stamps, but each stream independently drops a few frames (per-topic counts
differ). MCVO's pose_graph assumes the camera streams are in perfect
lockstep (same message count, same order) and segfaults otherwise, so only
timestamps present on ALL cameras are written.

Cameras run at 20 Hz; FRAME_STRIDE=2 keeps every 2nd synchronized triplet
(10 Hz) to keep the output bag size manageable (~38 GB instead of ~76 GB).

Run inside the MCVO Apptainer image with ROS sourced, e.g.:
  apptainer exec /gpfs/data/bkimia/cchien3/mcvo.simg bash -c \
    'source /opt/ros/noetic/setup.bash; python3 spires_to_mcvo_bag.py'
"""
from __future__ import print_function

import os
import sys


IN_BAG = ("/gpfs/data/bkimia/Datasets/Oxford_Spires/Observatory-Quarter-02/"
          "rosbag/1710338490_2024-03-13-14-01-30_0.bag")
OUT_BAG = ("/gpfs/data/bkimia/cchien3/MCVO-forked/MCVO/rosbags/"
           "OxfordSpires_ObsQuarter02.bag")
FRAME_STRIDE = 2  # keep every Nth frame per camera (2 -> 10 Hz)

TOPIC_MAP = {
    "/alphasense_driver_ros/cam0/debayered/image/compressed": ("/camera_00", "camera_00"),
    "/alphasense_driver_ros/cam1/debayered/image/compressed": ("/camera_01", "camera_01"),
    "/alphasense_driver_ros/cam2/debayered/image/compressed": ("/camera_02", "camera_02"),
}


def main():
    try:
        import cv2
        import numpy as np
        import rosbag
        from sensor_msgs.msg import Image
    except ImportError as e:
        sys.stderr.write(
            "Missing ROS/OpenCV Python module: {}\n"
            "Run this inside the MCVO Apptainer image with ROS sourced.\n".format(e))
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_BAG), exist_ok=True)

    print("Reading :", IN_BAG)
    print("Writing :", OUT_BAG)
    print("Stride  :", FRAME_STRIDE)

    # Pass 1 (index only): find HEADER stamps present on ALL camera topics.
    # The bag receive time differs per topic (recorded live), but shared
    # frames carry identical header stamps, so sync must use the header.
    # raw=True skips full deserialization; the std_msgs/Header at the start
    # of the buffer is: uint32 seq, uint32 stamp.secs, uint32 stamp.nsecs.
    import struct

    def header_stamp_nsec(raw_buffer):
        secs, nsecs = struct.unpack_from("<II", raw_buffer, 4)
        return secs * 1000000000 + nsecs

    stamps_per_topic = {t: set() for t in TOPIC_MAP}
    with rosbag.Bag(IN_BAG, "r") as inbag:
        for topic, msg, t in inbag.read_messages(topics=list(TOPIC_MAP), raw=True):
            stamps_per_topic[topic].add(header_stamp_nsec(msg[1]))

    common = set.intersection(*stamps_per_topic.values())
    kept = sorted(common)[::FRAME_STRIDE]
    kept_set = set(kept)
    for topic in sorted(TOPIC_MAP):
        n = len(stamps_per_topic[topic])
        print("{}: {} msgs, {} common, {} kept after stride".format(
            topic, n, len(common), len(kept)))

    written = {t: 0 for t in TOPIC_MAP}
    with rosbag.Bag(IN_BAG, "r") as inbag, rosbag.Bag(OUT_BAG, "w") as outbag:
        total = inbag.get_message_count(topic_filters=list(TOPIC_MAP))
        done = 0
        for topic, msg, t in inbag.read_messages(topics=list(TOPIC_MAP)):
            done += 1
            if done % 500 == 0:
                print("processed {}/{} msgs".format(done, total), flush=True)
            if msg.header.stamp.to_nsec() not in kept_set:
                continue

            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode msg at {} on {}".format(t.to_sec(), topic))

            out_topic, frame_id = TOPIC_MAP[topic]
            out = Image()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = frame_id
            out.height, out.width = img.shape[:2]
            out.encoding = "bgr8"
            out.is_bigendian = 0
            out.step = out.width * 3
            out.data = img.tobytes()
            outbag.write(out_topic, out, msg.header.stamp)
            written[topic] += 1

    counts = set(written.values())
    for topic in sorted(TOPIC_MAP):
        print("{} -> {}: wrote {} frames".format(
            topic, TOPIC_MAP[topic][0], written[topic]))
    if len(counts) != 1:
        raise RuntimeError("Camera streams are NOT in lockstep: {}".format(written))
    print("Done (all cameras in lockstep):", OUT_BAG)


if __name__ == "__main__":
    main()
