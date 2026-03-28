"""Replay a recorded video through the tracker pipeline with full visualisation."""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from capture.video_capture import VideoCapture
from feedback.visual_overlay import VisualOverlay
from tracking.pixel_tracker import PixelTracker
from utils.config_manager import ConfigManager


def _load_logged_data(csv_path: str) -> List[Dict[str, str]]:
    """Load CSV log from the original session for comparison overlay."""
    rows: List[Dict[str, str]] = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded video through tracker")
    parser.add_argument("video", type=str, help="Path to recorded video file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to original session CSV for comparison overlay")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    config = ConfigManager(args.config)
    pcfg = config.get_section("pixel_tracker")
    pcfg["kalman"] = config.get_section("kalman")

    cap = VideoCapture(source=args.video)
    tracker = PixelTracker(pcfg)

    frame, _, _ = cap.read()
    if frame is None:
        print("ERROR: Cannot read video file.")
        return

    h, w = frame.shape[:2]
    overlay = VisualOverlay(w, h)

    logged_data = _load_logged_data(args.csv) if args.csv else []

    paused = False
    step_mode = False
    initialized = False

    # Delay between frames (ms) — adjusted by speed multiplier
    base_delay = int(33 / max(0.1, args.speed))

    print("Controls: SPACE=init tracker, P=pause, N=step, D=debug, +/-=speed, Q=quit")

    while True:
        if not paused or step_mode:
            frame, ts, fnum = cap.read()
            if frame is None:
                print("End of video.")
                break
            step_mode = False

        if not initialized:
            state = "NOT TRACKING"
            out = overlay.draw(frame, None, None, 0.0, None, state, {}, cap.get_fps(), "pixel")
        else:
            target_pos, error, confidence, debug_info = tracker.update(frame)
            velocity = tracker._kalman.get_velocity() if tracker.is_initialized() else None

            if confidence < 0.1:
                state = "TARGET LOST"
            else:
                state = "TRACKING"

            out = overlay.draw(
                frame, target_pos, error, confidence, velocity, state, debug_info,
                cap.get_fps(), "pixel", list(tracker.trajectory),
            )

        # Show frame number
        cv2.putText(out, f"Frame: {cap.frame_number}", (10, h - 35),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if paused:
            cv2.putText(out, "PAUSED", (w // 2 - 40, 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Replay", out)

        key = cv2.waitKey(0 if paused else base_delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            tracker.initialize(frame)
            initialized = True
        elif key == ord("p"):
            paused = not paused
        elif key == ord("n"):
            step_mode = True
        elif key == ord("d"):
            overlay.toggle_debug()
        elif key == ord("r"):
            tracker.reset()
            initialized = False
        elif key == ord("+") or key == ord("="):
            base_delay = max(1, base_delay - 5)
        elif key == ord("-"):
            base_delay = min(200, base_delay + 5)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
