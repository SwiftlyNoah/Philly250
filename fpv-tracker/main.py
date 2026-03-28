"""FPV Drone Tracking Assistant — main entry point.

Runs the capture → track → feedback loop at video frame-rate (~30 Hz).
"""

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

from capture.video_capture import VideoCapture
from feedback.audio_cues import AudioCues
from feedback.visual_overlay import VisualOverlay
from tracking.pixel_tracker import PixelTracker
from tracking.yolo_tracker import YOLOTracker
from tracking.tracker_base import TrackerBase
from utils.config_manager import ConfigManager
from utils.data_logger import DataLogger
from utils.tuning_gui import TuningGUI


# ── Mouse callback state ────────────────────────────────────────────
_mouse_click_pos: Optional[tuple] = None


def _mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
    global _mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_click_pos = (x, y)


# ── CLI ──────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FPV Drone Tracking Assistant")
    p.add_argument("--source", default="0",
                   help="Camera index (int) or path to video file")
    p.add_argument("--config", default="config.yaml",
                   help="Path to config YAML")
    p.add_argument("--no-audio", action="store_true",
                   help="Disable audio cues")
    p.add_argument("--record", type=str, default=None,
                   help="Record the session to a video file (path)")
    p.add_argument("--debug", action="store_true",
                   help="Start with debug view enabled")
    p.add_argument("--tuning", action="store_true",
                   help="Open the tuning-slider GUI")
    return p.parse_args()


# ── Main loop ────────────────────────────────────────────────────────
def main() -> None:
    global _mouse_click_pos

    args = _parse_args()

    # --- Load config ---
    config = ConfigManager(args.config)

    # --- Video capture ---
    source: object = args.source
    try:
        source = int(source)
    except ValueError:
        pass  # It's a file path string

    vcfg = config.get_section("video")
    cap = VideoCapture(
        source=source,
        width=vcfg.get("width", 720),
        height=vcfg.get("height", 480),
        crop_rect=vcfg.get("crop"),
        flip=vcfg.get("flip", False),
    )

    # Grab one frame to know dimensions
    first_frame, _, _ = cap.read()
    if first_frame is None:
        print("ERROR: Cannot read from video source.")
        sys.exit(1)
    frame_h, frame_w = first_frame.shape[:2]

    # --- Trackers ---
    pcfg = config.get_section("pixel_tracker")
    pcfg["kalman"] = config.get_section("kalman")

    ycfg = config.get_section("yolo")
    ycfg["kalman"] = config.get_section("kalman")

    pixel_tracker = PixelTracker(pcfg)
    yolo_tracker = YOLOTracker(ycfg)

    active_tracker: TrackerBase = pixel_tracker

    # --- Overlay ---
    overlay = VisualOverlay(frame_w, frame_h)
    if args.debug:
        overlay.toggle_debug()

    # --- Audio ---
    acfg = config.get_section("audio")
    if args.no_audio:
        acfg["enabled"] = False
    audio = AudioCues(acfg)
    audio.set_frame_size(frame_w, frame_h)

    # --- Logger ---
    logger: Optional[DataLogger] = None
    if config.get("logging", "enabled"):
        log_dir = config.get("logging", "output_dir") or "logs/"
        logger = DataLogger(log_dir)

    # --- Video recorder ---
    recorder: Optional[cv2.VideoWriter] = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        recorder = cv2.VideoWriter(args.record, fourcc, 30.0, (frame_w, frame_h))

    # --- Tuning GUI ---
    tuning: Optional[TuningGUI] = None
    if args.tuning:
        tuning = TuningGUI(config)
        tuning.open()

    # --- Display window ---
    win_name = "FPV Tracker"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, _mouse_callback)

    print("FPV Drone Tracking Assistant")
    print("----------------------------")
    print("SPACE  : Init tracker (centre)")
    print("Click  : Init tracker (mouse)")
    print("R      : Reset tracker")
    print("T      : Toggle pixel / YOLO")
    print("M      : Mute / unmute audio")
    print("+/-    : Adjust volume")
    print("S      : Screenshot")
    print("P      : Pause (video file)")
    print("D      : Toggle debug view")
    print("Q      : Quit")
    print()

    paused = False

    # ── MAIN LOOP ────────────────────────────────────────────────────
    while True:
        # 1. Grab frame
        if not paused:
            frame, timestamp, frame_number = cap.read()
            if frame is None:
                if isinstance(source, str):
                    print("End of video file.")
                    break
                # Live feed dropped — skip frame
                time.sleep(0.01)
                continue
        else:
            # When paused (video file), keep displaying the last frame
            timestamp = time.time()

        # Feed-loss detection (live mode only)
        if isinstance(source, int) and cap.feed_lost():
            cv2.putText(frame, "FEED LOST", (frame_w // 2 - 60, frame_h // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 2. Update tracker
        target_pos = None
        error = None
        confidence = 0.0
        velocity = None
        debug_info: dict = {}
        tracker_state = "NOT TRACKING"

        if active_tracker.is_initialized():
            target_pos, error, confidence, debug_info = active_tracker.update(frame)
            velocity = None
            if hasattr(active_tracker, "_kalman"):
                velocity = active_tracker._kalman.get_velocity()

            if confidence < 0.1:
                tracker_state = "TARGET LOST"
            else:
                tracker_state = "TRACKING"

        # 3. Tuning GUI hot-update
        if tuning is not None:
            vals = tuning.read()
            if vals and isinstance(active_tracker, PixelTracker):
                active_tracker.update_config(vals)

        # 4. Visual overlay
        trajectory = None
        if isinstance(active_tracker, PixelTracker):
            trajectory = active_tracker.trajectory

        annotated = overlay.draw(
            frame,
            target_pos,
            error,
            confidence,
            velocity,
            tracker_state,
            debug_info,
            fps=cap.get_fps(),
            tracker_name=active_tracker.name,
            trajectory=trajectory,
        )

        if paused:
            cv2.putText(annotated, "PAUSED", (frame_w // 2 - 50, 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 5. Audio cue
        if error is not None:
            audio.update(error[0], error[1], confidence)

        # 6. Display
        cv2.imshow(win_name, annotated)

        # 7. Record
        if recorder is not None:
            recorder.write(annotated)

        # 8. Log
        if logger is not None and active_tracker.is_initialized():
            num_feat = debug_info.get("num_features", 0)
            inlier = debug_info.get("inlier_ratio", 0.0)
            tx = target_pos[0] if target_pos else None
            ty = target_pos[1] if target_pos else None
            ex = error[0] if error else None
            ey = error[1] if error else None
            logger.log(
                frame_number=cap.frame_number,
                timestamp=timestamp,
                target_x=tx,
                target_y=ty,
                error_x=ex,
                error_y=ey,
                confidence=confidence,
                tracker_type=active_tracker.name,
                num_features=num_feat,
                inlier_ratio=inlier,
                fps=cap.get_fps(),
            )

        # 9. Keyboard events
        key = cv2.waitKey(1) & 0xFF

        # Mouse click → initialise at click position
        if _mouse_click_pos is not None:
            mx, my = _mouse_click_pos
            _mouse_click_pos = None
            active_tracker.initialize(frame, float(mx), float(my))
            print(f"Tracker initialised at ({mx}, {my})")

        if key == ord(" "):
            active_tracker.initialize(frame)
            print("Tracker initialised at frame centre")

        elif key == ord("r"):
            active_tracker.reset()
            print("Tracker reset")

        elif key == ord("t"):
            was_init = active_tracker.is_initialized()
            old_pos = active_tracker.target_pos
            active_tracker.reset()

            if isinstance(active_tracker, PixelTracker):
                active_tracker = yolo_tracker
                if not yolo_tracker.model_loaded:
                    print("WARNING: YOLO model not loaded — tracker will coast on Kalman only")
            else:
                active_tracker = pixel_tracker

            if was_init and old_pos is not None:
                active_tracker.initialize(frame, old_pos[0], old_pos[1])

            print(f"Switched to {active_tracker.name.upper()} tracker")

        elif key == ord("m"):
            audio.toggle_mute()
            print(f"Audio {'muted' if audio.is_muted else 'unmuted'}")

        elif key == ord("+") or key == ord("="):
            audio.adjust_volume(0.1)
            print("Volume up")

        elif key == ord("-"):
            audio.adjust_volume(-0.1)
            print("Volume down")

        elif key == ord("s"):
            ss_path = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(ss_path, annotated)
            print(f"Screenshot saved: {ss_path}")

        elif key == ord("p"):
            if isinstance(source, str):
                paused = not paused
                print("Paused" if paused else "Resumed")

        elif key == ord("d"):
            overlay.toggle_debug()
            print(f"Debug view {'ON' if overlay.debug_mode else 'OFF'}")

        elif key == ord("q"):
            break

    # ── Cleanup ──────────────────────────────────────────────────────
    if recorder is not None:
        recorder.release()
    if logger is not None:
        logger.close()
        print(f"Session log saved: {logger.path}")
    if tuning is not None:
        tuning.close()
    audio.shutdown()
    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye.")


if __name__ == "__main__":
    main()
