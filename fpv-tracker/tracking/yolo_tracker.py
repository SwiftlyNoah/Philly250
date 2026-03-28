"""V2 tracker — YOLO detection with Kalman filter smoothing.

Supports optional **SAHI-style sliced inference** for detecting very small /
distant drones that occupy only a few pixels.  When ``slice_inference`` is
enabled in the config, the frame is divided into overlapping tiles, each tile
is run through the YOLO model at full resolution, and the detections are
merged with NMS — dramatically improving recall on tiny objects.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .kalman_filter import TargetKalmanFilter
from .tracker_base import TrackerBase


class YOLOTracker(TrackerBase):
    """Uses a YOLOv8 model for target detection, fused with a Kalman filter.

    When ``slice_inference`` is enabled (recommended for small-object
    detection), each frame is split into overlapping tiles and inference
    is run per-tile, then results are merged.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self._model_path: str = config.get("model_path", "models/best.pt")
        self._conf_thresh: float = config.get("confidence_threshold", 0.5)
        self._imgsz: int = config.get("imgsz", 640)

        # Sliced / tiled inference settings (SAHI-style)
        slice_cfg = config.get("slice_inference", {})
        if not isinstance(slice_cfg, dict):
            slice_cfg = {}
        self._slice_enabled: bool = slice_cfg.get("enabled", False)
        self._slice_size: int = slice_cfg.get("slice_size", 640)
        self._slice_overlap: float = slice_cfg.get("overlap", 0.25)
        self._slice_nms_thresh: float = slice_cfg.get("nms_threshold", 0.5)

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

    def load_weights(self, weights_path: str) -> bool:
        """Load (or swap) YOLO weights at runtime.

        Parameters
        ----------
        weights_path : str
            Path to a ``.pt`` weights file (e.g. your own trained model).

        Returns
        -------
        bool
            True if the model was loaded successfully.
        """
        old_path = self._model_path
        self._model_path = weights_path
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            return True
        except Exception as exc:
            print(f"WARNING: Failed to load YOLO weights from {weights_path}: {exc}")
            self._model_path = old_path
            self._load_model()  # try to restore the previous model
            return False

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
        if self._slice_enabled:
            detections = self._sliced_detect(frame)
        else:
            detections = self._full_frame_detect(frame)
        inference_ms = (time.time() - t0) * 1000
        debug["inference_time_ms"] = inference_ms
        debug["slice_inference"] = self._slice_enabled

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

        if self._slice_enabled:
            detections = self._sliced_detect(frame)
        else:
            detections = self._full_frame_detect(frame)

        best_pos = None
        best_dist = float("inf")
        for det in detections:
            dist = np.hypot(det["cx"] - target_x, det["cy"] - target_y)
            if dist < best_dist:
                best_pos = (float(det["cx"]), float(det["cy"]))
                best_dist = dist
        return best_pos

    # ------------------------------------------------------------------
    # Sliced (SAHI-style) inference
    # ------------------------------------------------------------------

    def _compute_slices(
        self, frame_w: int, frame_h: int
    ) -> List[Tuple[int, int, int, int]]:
        """Return a list of ``(x1, y1, x2, y2)`` tile rectangles that cover
        the entire frame with the configured overlap."""
        step = int(self._slice_size * (1.0 - self._slice_overlap))
        step = max(step, 1)
        slices: List[Tuple[int, int, int, int]] = []
        y = 0
        while y < frame_h:
            x = 0
            y2 = min(y + self._slice_size, frame_h)
            while x < frame_w:
                x2 = min(x + self._slice_size, frame_w)
                slices.append((x, y, x2, y2))
                if x2 >= frame_w:
                    break
                x += step
            if y2 >= frame_h:
                break
            y += step
        return slices

    def _full_frame_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO on the full frame and return a list of detection dicts."""
        if self._model is None:
            return []
        results = self._model(frame, imgsz=self._imgsz, conf=self._conf_thresh, verbose=False)
        detections: List[Dict[str, Any]] = []
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
        return detections

    def _sliced_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO on overlapping tiles and merge with NMS.

        Each tile is run at ``self._imgsz`` resolution, so small objects
        that would be invisible in a downscaled full-frame pass become
        clearly visible inside their tile.
        """
        if self._model is None:
            return []

        h, w = frame.shape[:2]
        slices = self._compute_slices(w, h)

        all_dets: List[Dict[str, Any]] = []
        for x1, y1, x2, y2 in slices:
            tile = frame[y1:y2, x1:x2]
            results = self._model(tile, imgsz=self._imgsz, conf=self._conf_thresh, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    # Map tile-local coords back to full-frame coords
                    abs_x1 = xyxy[0] + x1
                    abs_y1 = xyxy[1] + y1
                    abs_x2 = xyxy[2] + x1
                    abs_y2 = xyxy[3] + y1
                    cx = (abs_x1 + abs_x2) / 2.0
                    cy = (abs_y1 + abs_y2) / 2.0
                    bw = abs_x2 - abs_x1
                    bh = abs_y2 - abs_y1
                    all_dets.append({
                        "cx": cx, "cy": cy,
                        "w": bw, "h": bh,
                        "conf": conf,
                        "xyxy": [abs_x1, abs_y1, abs_x2, abs_y2],
                    })

        # Merge overlapping detections via NMS
        return self._nms(all_dets, self._slice_nms_thresh)

    @staticmethod
    def _nms(
        detections: List[Dict[str, Any]], iou_threshold: float
    ) -> List[Dict[str, Any]]:
        """Simple greedy NMS on a list of detection dicts."""
        if not detections:
            return []

        # Sort by confidence (highest first)
        dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
        keep: List[Dict[str, Any]] = []

        while dets:
            best = dets.pop(0)
            keep.append(best)
            remaining: List[Dict[str, Any]] = []
            bx1, by1, bx2, by2 = best["xyxy"]
            b_area = (bx2 - bx1) * (by2 - by1)
            for det in dets:
                dx1, dy1, dx2, dy2 = det["xyxy"]
                ix1 = max(bx1, dx1)
                iy1 = max(by1, dy1)
                ix2 = min(bx2, dx2)
                iy2 = min(by2, dy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                d_area = (dx2 - dx1) * (dy2 - dy1)
                union = b_area + d_area - inter
                iou = inter / union if union > 0 else 0.0
                if iou < iou_threshold:
                    remaining.append(det)
            dets = remaining

        return keep
