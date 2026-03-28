"""Real-time parameter tuning GUI using OpenCV trackbar sliders."""

from typing import Any, Dict

import cv2


class TuningGUI:
    """Window with trackbar sliders for the most important tracker parameters."""

    _WINDOW_NAME = "Tuning"

    def __init__(self, config_manager: object) -> None:
        self._cm = config_manager
        self._open = False

        self._sliders: Dict[str, Dict[str, Any]] = {
            "diff_threshold": {"section": "pixel_tracker", "min": 0, "max": 100},
            "search_radius": {"section": "pixel_tracker", "min": 10, "max": 300},
            "min_blob_area": {"section": "pixel_tracker", "min": 0, "max": 50},
            "max_blob_area": {"section": "pixel_tracker", "min": 50, "max": 2000},
            "edge_margin": {"section": "pixel_tracker", "min": 0, "max": 100},
            "blur_kernel": {"section": "pixel_tracker", "min": 1, "max": 15},
            "conf_engage": {"section": "kalman", "min": 0, "max": 100},
            "conf_disengage": {"section": "kalman", "min": 0, "max": 100},
            "deadzone": {"section": "audio", "min": 0, "max": 100},
        }

    def open(self) -> None:
        """Create the slider window and populate trackbars."""
        cv2.namedWindow(self._WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._WINDOW_NAME, 400, 450)

        for name, meta in self._sliders.items():
            section = meta["section"]
            key = name
            # Map some slider names to actual config keys
            if name == "conf_engage":
                key = "confidence_engage"
            elif name == "conf_disengage":
                key = "confidence_disengage"
            elif name == "deadzone":
                key = "deadzone_pixels"

            current = self._cm.get(section, key)  # type: ignore[union-attr]
            if current is None:
                current = meta["min"]

            # For fractional values stored as 0-1, scale to 0-100 for the slider
            if name in ("conf_engage", "conf_disengage"):
                current = int(float(current) * 100)
            else:
                current = int(current)

            cv2.createTrackbar(name, self._WINDOW_NAME, current, meta["max"], lambda _v: None)

        self._open = True

    def read(self) -> Dict[str, Any]:
        """Read all trackbar positions and return as a flat dict."""
        if not self._open:
            return {}

        values: Dict[str, Any] = {}
        for name, meta in self._sliders.items():
            try:
                val = cv2.getTrackbarPos(name, self._WINDOW_NAME)
            except cv2.error:
                continue

            if name == "blur_kernel":
                val = val if val % 2 == 1 else val + 1
                val = max(1, val)

            if name in ("conf_engage", "conf_disengage"):
                values[name] = val / 100.0
            else:
                values[name] = val

        return values

    def close(self) -> None:
        if self._open:
            try:
                cv2.destroyWindow(self._WINDOW_NAME)
            except cv2.error:
                pass
            self._open = False
