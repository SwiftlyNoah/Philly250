"""Record video from the capture card for later labeling and training."""

import argparse
import os
import sys
import time

import cv2

# Allow running as a standalone script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from capture.video_capture import VideoCapture
from utils.config_manager import ConfigManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Record capture card feed to video file")
    parser.add_argument("--source", type=int, default=0, help="Camera device index")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--save-frames", type=int, default=0,
                        help="Save individual frames every N frames (0 = disabled)")
    parser.add_argument("--frames-dir", type=str, default="frames/",
                        help="Directory for saved individual frames")
    args = parser.parse_args()

    config = ConfigManager(args.config)
    vcfg = config.get_section("video")

    cap = VideoCapture(
        source=args.source,
        width=vcfg.get("width", 720),
        height=vcfg.get("height", 480),
        crop_rect=vcfg.get("crop"),
        flip=vcfg.get("flip", False),
    )

    # Determine output path
    if args.output is None:
        os.makedirs("recordings", exist_ok=True)
        args.output = os.path.join("recordings", f"session_{time.strftime('%Y%m%d_%H%M%S')}.avi")

    # Read one frame to get dimensions
    frame, _, _ = cap.read()
    if frame is None:
        print("ERROR: Cannot read from video source.")
        return

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(args.output, fourcc, 30.0, (w, h))
    print(f"Recording to {args.output}  ({w}x{h})  —  press Q to stop")

    if args.save_frames > 0:
        os.makedirs(args.frames_dir, exist_ok=True)

    frame_count = 0
    while True:
        frame, ts, fnum = cap.read()
        if frame is None:
            continue

        writer.write(frame)
        frame_count += 1

        if args.save_frames > 0 and frame_count % args.save_frames == 0:
            path = os.path.join(args.frames_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(path, frame)

        # Show preview
        cv2.putText(frame, f"REC  frame {frame_count}", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Recording — press Q to stop", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
