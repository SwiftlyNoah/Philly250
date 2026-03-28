"""Semi-automated labeling tool — uses the pixel tracker to bootstrap YOLO labels."""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from capture.video_capture import VideoCapture
from tracking.pixel_tracker import PixelTracker
from utils.config_manager import ConfigManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Semi-auto labeling with pixel tracker")
    parser.add_argument("video", type=str, help="Path to recorded video file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config YAML path")
    parser.add_argument("--output-dir", type=str, default="dataset/", help="Output directory for labels + images")
    parser.add_argument("--bbox-size", type=int, default=40, help="Default bounding box side length (pixels)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum tracker confidence to auto-propose a label")
    parser.add_argument("--every-n", type=int, default=5, help="Label every Nth frame")
    args = parser.parse_args()

    config = ConfigManager(args.config)
    pcfg = config.get_section("pixel_tracker")
    pcfg["kalman"] = config.get_section("kalman")

    images_dir = os.path.join(args.output_dir, "images")
    labels_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cap = VideoCapture(source=args.video)
    tracker = PixelTracker(pcfg)

    frame, _, _ = cap.read()
    if frame is None:
        print("ERROR: Cannot read video file.")
        return

    h, w = frame.shape[:2]
    half = args.bbox_size // 2
    initialized = False
    saved_count = 0
    skipped = 0

    print("Controls: SPACE=init tracker, A=accept, S=skip, E=edit bbox, Q=quit")

    while True:
        frame, ts, fnum = cap.read()
        if frame is None:
            print(f"End of video. Saved {saved_count} labels, skipped {skipped}.")
            break

        if not initialized:
            display = frame.copy()
            cv2.putText(display, "Press SPACE to init tracker at centre", (10, 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Label Tool", display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                tracker.initialize(frame)
                initialized = True
            elif key == ord("q"):
                break
            continue

        target_pos, error, confidence, debug_info = tracker.update(frame)

        # Only propose labels on every Nth frame with sufficient confidence
        if fnum % args.every_n != 0:
            continue

        if target_pos is None or confidence < args.min_confidence:
            continue

        tx, ty = int(target_pos[0]), int(target_pos[1])

        # Draw proposed bbox
        display = frame.copy()
        x1 = max(0, tx - half)
        y1 = max(0, ty - half)
        x2 = min(w, tx + half)
        y2 = min(h, ty + half)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(display, (tx, ty), 3, (0, 255, 0), -1)
        cv2.putText(display, f"Conf: {confidence:.2f}  Frame: {fnum}",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "[A]ccept  [S]kip  [E]dit  [Q]uit",
                     (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow("Label Tool", display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            skipped += 1
            continue
        elif key == ord("e"):
            # Let user draw a custom bbox
            roi = cv2.selectROI("Edit BBox", frame, showCrosshair=True)
            cv2.destroyWindow("Edit BBox")
            if roi[2] > 0 and roi[3] > 0:
                x1, y1 = int(roi[0]), int(roi[1])
                x2, y2 = x1 + int(roi[2]), y1 + int(roi[3])
            else:
                skipped += 1
                continue
        elif key == ord("a"):
            pass  # Accept the proposed bbox as-is
        else:
            continue

        # Save image + YOLO-format label
        img_name = f"frame_{fnum:06d}.jpg"
        cv2.imwrite(os.path.join(images_dir, img_name), frame)

        # YOLO format: class x_center y_center width height (normalised)
        cx_norm = ((x1 + x2) / 2.0) / w
        cy_norm = ((y1 + y2) / 2.0) / h
        bw_norm = (x2 - x1) / w
        bh_norm = (y2 - y1) / h
        label_name = f"frame_{fnum:06d}.txt"
        with open(os.path.join(labels_dir, label_name), "w") as fh:
            fh.write(f"0 {cx_norm:.6f} {cy_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

        saved_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {saved_count} labels saved to {args.output_dir}")


if __name__ == "__main__":
    main()
