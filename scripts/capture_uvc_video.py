#!/usr/bin/env python3
"""Capture RGB video from any UVC camera via OpenCV.

This is the simplest way to test an Intel RealSense D435/D435i in RGB-only mode
without depending on pyrealsense2. On Linux, the RGB sensor is often exposed as
one of the regular ``/dev/video*`` devices.
"""

from __future__ import annotations

import argparse
import time

import cv2

from reifyscene.common import data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--output", required=True,
                        help="the .mp4 to write, by its name in the data directory's datasets/")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seconds", type=float, default=10.0, help="Capture duration")
    parser.add_argument("--show", action="store_true", help="Show a preview window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = data_dir.path(data_dir.DATASETS, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.device}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer for {args.output}.")

    start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = cv2.resize(frame, (args.width, args.height))
            writer.write(frame)
            if args.show:
                cv2.imshow("capture_uvc_video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if (time.time() - start) >= args.seconds:
                break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Saved RGB clip to: {args.output}")


if __name__ == "__main__":
    main()
