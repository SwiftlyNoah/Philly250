"""V1 tracker — ego-motion compensation via optical flow + residual blob detection."""

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .kalman_filter import TargetKalmanFilter
from .tracker_base import TrackerBase


class PixelTracker(TrackerBase):
    """Background-subtraction tracker that compensates for ego-motion using
    sparse optical flow and homography estimation, then detects the target
    as the residual moving blob.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        # Feature detection
        self._feature_count: int = config.get("feature_count", 300)
        self._feature_quality: float = config.get("feature_quality", 0.01)
        self._feature_min_dist: int = config.get("feature_min_distance", 10)

        # Optical flow (Lucas-Kanade)
        self._lk_params = dict(
            winSize=(config.get("lk_window_size", 21), config.get("lk_window_size", 21)),
            maxLevel=config.get("lk_pyramid_levels", 3),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Homography
        self._ransac_thresh: float = config.get("ransac_threshold", 3.0)
        self._min_inlier_ratio: float = config.get("min_inlier_ratio", 0.5)

        # Differencing
        self._diff_threshold: int = config.get("diff_threshold", 25)
        self._blur_kernel: int = config.get("blur_kernel", 5)
        self._morph_kernel: int = config.get("morph_kernel", 3)

        # Blob filtering
        self._min_blob_area: int = config.get("min_blob_area", 1)
        self._max_blob_area: int = config.get("max_blob_area", 500)
        self._search_radius: int = config.get("search_radius", 100)
        self._edge_margin: int = config.get("edge_margin", 40)
        self._target_exclusion_radius: int = config.get("target_exclusion_radius", 50)

        # Kalman filter (will be created on initialize)
        kalman_cfg = config.get("kalman", {})
        if not isinstance(kalman_cfg, dict):
            kalman_cfg = {}
        self._kalman = TargetKalmanFilter(
            process_noise=kalman_cfg.get("process_noise", 0.01),
            measurement_noise=kalman_cfg.get("measurement_noise", 0.1),
        )
        self._max_frames_lost: int = kalman_cfg.get("max_frames_lost", 30)
        self._conf_engage: float = kalman_cfg.get("confidence_engage", 0.7)
        self._conf_disengage: float = kalman_cfg.get("confidence_disengage", 0.2)

        # Internal state
        self._prev_gray: Optional[np.ndarray] = None
        self._frames_since_detection: int = 0
        self._trajectory: Deque[Tuple[float, float]] = deque(maxlen=200)
        self._frame_center: Tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # TrackerBase interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "pixel"

    def initialize(
        self, frame: np.ndarray, target_x: Optional[float] = None, target_y: Optional[float] = None
    ) -> None:
        h, w = frame.shape[:2]
        self._frame_center = (w / 2.0, h / 2.0)

        if target_x is None:
            target_x = self._frame_center[0]
        if target_y is None:
            target_y = self._frame_center[1]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        self._prev_gray = gray

        self._target_pos = (float(target_x), float(target_y))
        self._kalman.initialize(float(target_x), float(target_y))
        self._confidence = 1.0
        self._frames_since_detection = 0
        self._trajectory.clear()
        self._trajectory.append(self._target_pos)
        self._initialized = True

    def update(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], float, Dict[str, Any]]:
        if not self._initialized or self._prev_gray is None:
            return None, None, 0.0, {}

        h, w = frame.shape[:2]
        self._frame_center = (w / 2.0, h / 2.0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

        debug: Dict[str, Any] = {
            "diff_image": None,
            "thresh_image": None,
            "num_features": 0,
            "num_tracked": 0,
            "inlier_ratio": 0.0,
            "num_blobs": 0,
            "detection_found": False,
            "search_center": None,
            "all_blobs": [],
        }

        # --- Part 1: Background feature detection ---
        mask = self._build_feature_mask(gray, self._target_pos)
        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=self._feature_count,
            qualityLevel=self._feature_quality,
            minDistance=self._feature_min_dist,
            mask=mask,
        )
        if prev_pts is None or len(prev_pts) < 8:
            debug["num_features"] = 0 if prev_pts is None else len(prev_pts)
            return self._kalman_coast(gray, debug)

        debug["num_features"] = len(prev_pts)

        # --- Part 2: Optical flow ---
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None, **self._lk_params
        )
        if next_pts is None:
            return self._kalman_coast(gray, debug)

        good_mask = status.ravel() == 1
        old_pts = prev_pts[good_mask]
        new_pts = next_pts[good_mask]

        if len(old_pts) < 8:
            debug["num_tracked"] = len(old_pts)
            return self._kalman_coast(gray, debug)

        debug["num_tracked"] = len(old_pts)

        # --- Part 3: Homography ---
        H, inliers_mask = cv2.findHomography(old_pts, new_pts, cv2.RANSAC, self._ransac_thresh)
        if H is None or inliers_mask is None:
            return self._kalman_coast(gray, debug)

        inlier_ratio = float(np.sum(inliers_mask)) / len(inliers_mask) if len(inliers_mask) > 0 else 0.0
        debug["inlier_ratio"] = inlier_ratio

        if inlier_ratio < self._min_inlier_ratio:
            return self._kalman_coast(gray, debug)

        # --- Part 4: Frame warping and differencing ---
        warped_prev = cv2.warpPerspective(self._prev_gray, H, (w, h))
        diff = cv2.absdiff(warped_prev, gray)

        bk = self._blur_kernel
        if bk % 2 == 0:
            bk += 1
        diff = cv2.GaussianBlur(diff, (bk, bk), 0)
        _, thresh = cv2.threshold(diff, self._diff_threshold, 255, cv2.THRESH_BINARY)

        mk = self._morph_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Zero out edge borders
        em = self._edge_margin
        if em > 0:
            thresh[:em, :] = 0
            thresh[-em:, :] = 0
            thresh[:, :em] = 0
            thresh[:, -em:] = 0

        debug["diff_image"] = diff
        debug["thresh_image"] = thresh

        # --- Part 5: Blob detection ---
        pred_x, pred_y = self._kalman.predict()
        debug["search_center"] = (pred_x, pred_y)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[float, float, float]] = []  # (cx, cy, area)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_blob_area or area > self._max_blob_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            candidates.append((cx, cy, area))

        debug["num_blobs"] = len(candidates)
        debug["all_blobs"] = candidates

        # Find the candidate closest to the Kalman prediction within search radius
        best: Optional[Tuple[float, float]] = None
        best_dist = float("inf")
        for cx, cy, _ in candidates:
            dist = np.hypot(cx - pred_x, cy - pred_y)
            if dist < self._search_radius and dist < best_dist:
                best = (cx, cy)
                best_dist = dist

        # --- Part 6: Kalman update ---
        if best is not None:
            corrected = self._kalman.correct(best[0], best[1])
            self._target_pos = corrected
            self._frames_since_detection = 0
            self._confidence = min(1.0, self._confidence + 0.1)
            debug["detection_found"] = True
        else:
            self._target_pos = (pred_x, pred_y)
            self._frames_since_detection += 1
            self._confidence = max(0.0, self._confidence - 0.05)

        # Check for tracking loss
        if self._frames_since_detection > self._max_frames_lost:
            self._confidence = 0.0

        if self._target_pos is not None:
            tx, ty = self._target_pos
            if tx < 0 or tx >= w or ty < 0 or ty >= h:
                self._confidence = 0.0

        # --- Part 7: Prepare output ---
        self._trajectory.append(self._target_pos)
        self._prev_gray = gray

        error: Optional[Tuple[float, float]] = None
        if self._target_pos is not None:
            error = (
                self._target_pos[0] - self._frame_center[0],
                self._target_pos[1] - self._frame_center[1],
            )

        return self._target_pos, error, self._confidence, debug

    def reset(self) -> None:
        self._prev_gray = None
        self._target_pos = None
        self._confidence = 0.0
        self._initialized = False
        self._frames_since_detection = 0
        self._trajectory.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_feature_mask(
        self, gray: np.ndarray, target_pos: Optional[Tuple[float, float]]
    ) -> np.ndarray:
        """Create a mask that excludes the target region and frame edges."""
        h, w = gray.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        em = self._edge_margin
        if em > 0:
            mask[:em, :] = 0
            mask[-em:, :] = 0
            mask[:, :em] = 0
            mask[:, -em:] = 0

        if target_pos is not None:
            tx, ty = int(target_pos[0]), int(target_pos[1])
            cv2.circle(mask, (tx, ty), self._target_exclusion_radius, 0, -1)

        return mask

    def _kalman_coast(
        self, gray: np.ndarray, debug: Dict[str, Any]
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], float, Dict[str, Any]]:
        """Fall back to Kalman prediction only (no measurement)."""
        pred_x, pred_y = self._kalman.predict()
        self._target_pos = (pred_x, pred_y)
        self._frames_since_detection += 1
        self._confidence = max(0.0, self._confidence - 0.05)

        if self._frames_since_detection > self._max_frames_lost:
            self._confidence = 0.0

        self._trajectory.append(self._target_pos)
        self._prev_gray = gray

        error: Optional[Tuple[float, float]] = None
        if self._target_pos is not None:
            error = (
                self._target_pos[0] - self._frame_center[0],
                self._target_pos[1] - self._frame_center[1],
            )

        return self._target_pos, error, self._confidence, debug

    @property
    def trajectory(self) -> list:
        return list(self._trajectory)

    def update_config(self, config: Dict[str, Any]) -> None:
        """Hot-update tunable parameters from the tuning GUI."""
        if "diff_threshold" in config:
            self._diff_threshold = int(config["diff_threshold"])
        if "search_radius" in config:
            self._search_radius = int(config["search_radius"])
        if "min_blob_area" in config:
            self._min_blob_area = int(config["min_blob_area"])
        if "max_blob_area" in config:
            self._max_blob_area = int(config["max_blob_area"])
        if "edge_margin" in config:
            self._edge_margin = int(config["edge_margin"])
        if "blur_kernel" in config:
            bk = int(config["blur_kernel"])
            self._blur_kernel = bk if bk % 2 == 1 else bk + 1
