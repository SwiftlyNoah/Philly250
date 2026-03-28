"""V2 tracker — YOLO detection with Kalman filter smoothing."""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .kalman_filter import TargetKalmanFilter
from .tracker_base import TrackerBase


class YOLOTracker(TrackerBase):
    """Uses a YOLOv8 model for target detection, fused with a Kalman filter."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self._model_path: str = config.get("model_path", "models/best.pt")
        self._conf_thresh: float = config.get("confidence_threshold", 0.5)
        self._imgsz: int = config.get("imgsz", 640)

        kalman_cfg = config.get("kalman", {})
        if not isinstance(kalman_cfg, dict):
            kalman_cfg = {}
        self._kalman = TargetKalmanFilter(
            process_noise=kalman_cfg.get("process_noise", 0.01),
            measurement_noise=kalman_cfg.get("measurement_noise", 0.1),
        )
        self._max_frames_lost: int = kalman_cfg.get("max_frames_lost", 30)

        self._model: object = None
        self._frames_since_detection: int = 0
        self._frame_center: Tuple[float, float] = (0.0, 0.0)
        self._search_radius: int = config.get("search_radius", 200)

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the YOLO model. Fail gracefully if unavailable."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
        except Exception:
            self._model = None

    @property
    def name(self) -> str:
        return "yolo"

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def initialize(
        self, frame: np.ndarray, target_x: Optional[float] = None, target_y: Optional[float] = None
    ) -> None:
        h, w = frame.shape[:2]
        self._frame_center = (w / 2.0, h / 2.0)

        if target_x is None:
            target_x = self._frame_center[0]
        if target_y is None:
            target_y = self._frame_center[1]

        self._target_pos = (float(target_x), float(target_y))
        self._kalman.initialize(float(target_x), float(target_y))
        self._confidence = 1.0
        self._frames_since_detection = 0
        self._initialized = True

        # Try to find a YOLO detection near the target
        if self._model is not None:
            det = self._detect_nearest(frame, target_x, target_y)
            if det is not None:
                self._kalman.initialize(det[0], det[1])
                self._target_pos = det

    def update(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], float, Dict[str, Any]]:
        if not self._initialized:
            return None, None, 0.0, {}

        h, w = frame.shape[:2]
        self._frame_center = (w / 2.0, h / 2.0)

        debug: Dict[str, Any] = {
            "all_detections": [],
            "selected_detection": None,
            "inference_time_ms": 0.0,
        }

        if self._model is None:
            # No model loaded — coast on Kalman
            pred = self._kalman.predict()
            self._target_pos = pred
            self._frames_since_detection += 1
            self._confidence = max(0.0, self._confidence - 0.05)
            if self._frames_since_detection > self._max_frames_lost:
                self._confidence = 0.0
            error = (
                self._target_pos[0] - self._frame_center[0],
                self._target_pos[1] - self._frame_center[1],
            )
            return self._target_pos, error, self._confidence, debug

        t0 = time.time()
        results = self._model(frame, imgsz=self._imgsz, conf=self._conf_thresh, verbose=False)
        inference_ms = (time.time() - t0) * 1000
        debug["inference_time_ms"] = inference_ms

        # Collect all detections
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                detections.append({
                    "cx": cx, "cy": cy,
                    "w": bw, "h": bh,
                    "conf": conf,
                    "xyxy": xyxy.tolist(),
                })

        debug["all_detections"] = detections

        # Associate: pick the detection closest to Kalman prediction
        pred_x, pred_y = self._kalman.predict()
        best = None
        best_dist = float("inf")
        for det in detections:
            dist = np.hypot(det["cx"] - pred_x, det["cy"] - pred_y)
            if dist < self._search_radius and dist < best_dist:
                best = det
                best_dist = dist

        if best is not None:
            corrected = self._kalman.correct(best["cx"], best["cy"])
            self._target_pos = corrected
            self._frames_since_detection = 0
            self._confidence = min(1.0, self._confidence + 0.1)
            debug["selected_detection"] = best
        else:
            self._target_pos = (pred_x, pred_y)
            self._frames_since_detection += 1
            self._confidence = max(0.0, self._confidence - 0.05)

        if self._frames_since_detection > self._max_frames_lost:
            self._confidence = 0.0

        error = (
            self._target_pos[0] - self._frame_center[0],
            self._target_pos[1] - self._frame_center[1],
        )
        return self._target_pos, error, self._confidence, debug

    def reset(self) -> None:
        self._target_pos = None
        self._confidence = 0.0
        self._initialized = False
        self._frames_since_detection = 0

    def _detect_nearest(
        self, frame: np.ndarray, target_x: float, target_y: float
    ) -> Optional[Tuple[float, float]]:
        """Run detection and return the centre of the detection nearest to (target_x, target_y)."""
        if self._model is None:
            return None
        results = self._model(frame, imgsz=self._imgsz, conf=self._conf_thresh, verbose=False)
        best_pos = None
        best_dist = float("inf")
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                dist = np.hypot(cx - target_x, cy - target_y)
                if dist < best_dist:
                    best_pos = (float(cx), float(cy))
                    best_dist = dist
        return best_pos
