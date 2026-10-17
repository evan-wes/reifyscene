#!/usr/bin/env python3
"""Capture RGB video from an Intel RealSense camera using pyrealsense2.

This script is optional. For early RGB-only testing, ``capture_uvc_video.py`` is
usually the lower-friction option because it does not require the RealSense SDK.
"""

from __future__ import annotations

import argparse
import time

import cv2

from reifyscene.common import data_dir

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyrealsense2 is not installed. Run scripts/setup_realsense_sdk.sh or use scripts/capture_uvc_video.py instead."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="the .mp4 to write, by its name in the data directory's datasets/")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = data_dir.path(data_dir.DATASETS, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        pipeline.stop()
        raise RuntimeError(f"Could not open output writer for {args.output}.")

    print(f"Started RealSense RGB capture on device: {profile.get_device().get_info(rs.camera_info.name)}")
    start = time.time()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            import numpy as np
            frame = np.asanyarray(color.get_data())
            frame = cv2.resize(frame, (args.width, args.height))
            writer.write(frame)
            if args.show:
                cv2.imshow("capture_realsense_rgb", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if (time.time() - start) >= args.seconds:
                break
    finally:
        pipeline.stop()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Saved RGB clip to: {args.output}")


if __name__ == "__main__":
    main()
