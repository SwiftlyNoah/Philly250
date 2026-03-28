"""Video capture module — grabs frames from a capture card or video file."""

import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


class VideoCapture:
    """Wraps OpenCV VideoCapture with crop, flip, FPS tracking, and feed-loss detection."""

    def __init__(
        self,
        source: object = 0,
        width: int = 720,
        height: int = 480,
        crop_rect: Optional[List[int]] = None,
        flip: bool = False,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._crop_rect = crop_rect  # [x, y, w, h]
        self._flip = flip

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_number: int = 0
        self._timestamps: Deque[float] = deque(maxlen=60)
        self._last_good_time: float = 0.0

        self._open(source)

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------
    def _open(self, source: object) -> None:
        """Open the video source, trying multiple device indices if needed."""
        if isinstance(source, str):
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {source}")
            return

        # For integer device index, try the given index first, then 0-4
        indices_to_try = [int(source)]
        for i in range(5):
            if i not in indices_to_try:
                indices_to_try.append(i)

        for idx in indices_to_try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._cap = cap
                    self._configure_resolution()
                    return
                cap.release()

        raise RuntimeError(
            f"Cannot open any video device (tried indices {indices_to_try})"
        )

    def _configure_resolution(self) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------
    def read(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Return ``(frame, timestamp, frame_number)``."""
        if self._cap is None or not self._cap.isOpened():
            return None, time.time(), self._frame_number

        ret, frame = self._cap.read()
        now = time.time()

        if not ret or frame is None:
            return None, now, self._frame_number

        self._frame_number += 1
        self._last_good_time = now
        self._timestamps.append(now)

        # Handle YUV capture cards
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if self._flip:
            frame = cv2.flip(frame, -1)

        if self._crop_rect is not None:
            x, y, w, h = self._crop_rect
            frame = frame[y : y + h, x : x + w]

        return frame, now, self._frame_number

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_fps(self) -> float:
        """Return rolling-average FPS computed from recent frame timestamps."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def feed_lost(self, timeout: float = 2.0) -> bool:
        """Return True if no good frame has arrived within *timeout* seconds."""
        if self._last_good_time == 0.0:
            return False
        return (time.time() - self._last_good_time) > timeout

    def calibrate_crop(self) -> Optional[List[int]]:
        """Interactive crop calibration — let the user draw a rectangle on the raw feed.

        Returns the selected ``[x, y, w, h]`` or ``None`` if cancelled.
        """
        if self._cap is None:
            return None

        # Temporarily disable crop to show the full frame
        old_crop = self._crop_rect
        self._crop_rect = None
        frame, _, _ = self.read()
        self._crop_rect = old_crop

        if frame is None:
            return None

        roi = cv2.selectROI("Calibrate Crop (press ENTER to confirm, ESC to cancel)", frame, showCrosshair=True)
        cv2.destroyWindow("Calibrate Crop (press ENTER to confirm, ESC to cancel)")

        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        if w == 0 or h == 0:
            return None

        self._crop_rect = [x, y, w, h]
        return self._crop_rect

    @property
    def frame_number(self) -> int:
        return self._frame_number

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
