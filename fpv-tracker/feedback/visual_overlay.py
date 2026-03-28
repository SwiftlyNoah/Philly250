"""HUD overlay — reticles, arrows, graphs, and debug insets."""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _lerp_color(
    c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float
) -> Tuple[int, int, int]:
    """Linearly interpolate between two BGR colours (t=0 → c1, t=1 → c2)."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


# Colour constants (BGR)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GRAY = (160, 160, 160)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
DIM_GREEN = (0, 140, 0)

# Zone thresholds (normalised distance from centre: 0 = centre, 1 = edge).
# Green  = small central area  (on-target)
# Yellow = medium warning band
# Red    = large edge area      (losing target)
_ZONE_GREEN_MAX = 0.35   # green when norm <= 0.35
_ZONE_YELLOW_MAX = 0.65  # yellow when 0.35 < norm < 0.65, red beyond


def get_zone(
    error_x: float,
    error_y: float,
    half_w: float,
    half_h: float,
) -> tuple:
    """Return (colour_bgr, label, norm_distance) based on target offset.

    *norm_distance* is max(|ex|/half_w, |ey|/half_h) so it equals 1.0 when
    the target sits right on the frame edge.
    """
    norm = max(abs(error_x) / half_w, abs(error_y) / half_h) if half_w and half_h else 0.0
    if norm < _ZONE_GREEN_MAX:
        return GREEN, "TRACKING", norm
    if norm < _ZONE_YELLOW_MAX:
        return YELLOW, "WARNING", norm
    return RED, "CRITICAL", norm


class VisualOverlay:
    """Draws all HUD elements onto the display frame."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self._w = frame_width
        self._h = frame_height
        self._cx = frame_width / 2.0
        self._cy = frame_height / 2.0
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._debug_mode = False

    def toggle_debug(self) -> None:
        self._debug_mode = not self._debug_mode

    @property
    def debug_mode(self) -> bool:
        return self._debug_mode

    def resize(self, frame_width: int, frame_height: int) -> None:
        self._w = frame_width
        self._h = frame_height
        self._cx = frame_width / 2.0
        self._cy = frame_height / 2.0

    def draw(
        self,
        frame: np.ndarray,
        target_pos: Optional[Tuple[float, float]],
        error: Optional[Tuple[float, float]],
        confidence: float,
        velocity: Optional[Tuple[float, float]],
        tracker_state: str,
        debug_info: Dict[str, Any],
        fps: float = 0.0,
        tracker_name: str = "pixel",
        trajectory: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Render all HUD elements and return the annotated frame."""
        out = frame.copy()
        h, w = out.shape[:2]
        if w != self._w or h != self._h:
            self.resize(w, h)

        cx_i, cy_i = int(self._cx), int(self._cy)

        # --- Always visible ---
        # Crosshair at frame centre
        cv2.line(out, (cx_i - 20, cy_i), (cx_i + 20, cy_i), GRAY, 1)
        cv2.line(out, (cx_i, cy_i - 20), (cx_i, cy_i + 20), GRAY, 1)

        # State label
        state_colors = {
            "NOT TRACKING": GRAY,
            "TRACKING": GREEN,
            "TARGET LOST": RED,
        }
        state_col = state_colors.get(tracker_state, GRAY)
        cv2.putText(out, tracker_state, (10, 25), self._font, 0.6, state_col, 2)

        # FPS + tracker type
        cv2.putText(out, f"FPS: {fps:.0f}", (10, h - 15), self._font, 0.45, WHITE, 1)
        cv2.putText(out, tracker_name.upper(), (w - 80, 25), self._font, 0.5, CYAN, 1)

        if target_pos is None or tracker_state == "NOT TRACKING":
            return out

        tx, ty = int(target_pos[0]), int(target_pos[1])

        # Zone-based colour (position from centre)
        if error is not None:
            zone_col, zone_label, zone_norm = get_zone(
                error[0], error[1], self._cx, self._cy,
            )
        else:
            zone_col, zone_label, zone_norm = GREEN, "TRACKING", 0.0

        col = zone_col

        # --- Tracking HUD ---
        # Target reticle
        radius = 18
        cv2.circle(out, (tx, ty), radius, col, 2)
        cv2.circle(out, (tx, ty), 3, col, -1)

        # Error line
        if error is not None:
            cv2.line(out, (cx_i, cy_i), (tx, ty), col, 1, cv2.LINE_AA)

        # Zone label near target reticle
        if error is not None:
            cv2.putText(
                out, zone_label, (tx - 20, ty - 25),
                self._font, 0.45, zone_col, 1,
            )

        # Direction arrow (primary guidance element)
        if error is not None:
            ex, ey = error
            mag = math.hypot(ex, ey)
            if mag > 5:
                nx, ny = ex / mag, ey / mag
                arrow_len = min(mag * 0.6, 120)
                ax = int(self._cx + nx * arrow_len)
                ay = int(self._cy + ny * arrow_len)
                cv2.arrowedLine(out, (cx_i, cy_i), (ax, ay), zone_col, 3, tipLength=0.3)

        # Lead indicator
        if velocity is not None:
            vx, vy = velocity
            speed = np.hypot(vx, vy)
            if speed > 0.5:
                lead_frames = 10
                lx = int(target_pos[0] + vx * lead_frames)
                ly = int(target_pos[1] + vy * lead_frames)
                cv2.circle(out, (lx, ly), 12, DIM_GREEN, 1, cv2.LINE_AA)
                cv2.line(out, (tx, ty), (lx, ly), DIM_GREEN, 1, cv2.LINE_AA)

        # Trajectory trail
        if trajectory and len(trajectory) > 1:
            pts = trajectory
            n = len(pts)
            for i in range(1, n):
                alpha = i / n
                trail_col = _lerp_color((60, 60, 60), col, alpha)
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(out, p1, p2, trail_col, 1, cv2.LINE_AA)

        # Error bar graphs (edge bars)
        if error is not None:
            ex, ey = error
            bar_max = max(self._w, self._h) / 2
            # Horizontal error bar (bottom)
            bx_len = int(min(abs(ex) / bar_max, 1.0) * (w // 4))
            bx_start = w // 2
            if ex > 0:
                cv2.rectangle(out, (bx_start, h - 8), (bx_start + bx_len, h - 2), col, -1)
            else:
                cv2.rectangle(out, (bx_start - bx_len, h - 8), (bx_start, h - 2), col, -1)
            # Vertical error bar (right edge)
            by_len = int(min(abs(ey) / bar_max, 1.0) * (h // 4))
            by_start = h // 2
            if ey > 0:
                cv2.rectangle(out, (w - 8, by_start), (w - 2, by_start + by_len), col, -1)
            else:
                cv2.rectangle(out, (w - 8, by_start - by_len), (w - 2, by_start), col, -1)

        # Distance-from-centre ring
        if error is not None:
            ring_r = int(np.hypot(error[0], error[1]))
            if ring_r > 5:
                overlay = out.copy()
                cv2.circle(overlay, (cx_i, cy_i), ring_r, col, 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

        # Confidence meter (top-right corner bar)
        bar_w, bar_h = 10, 80
        bx0, by0 = w - 25, 40
        cv2.rectangle(out, (bx0, by0), (bx0 + bar_w, by0 + bar_h), GRAY, 1)
        fill = int(bar_h * confidence)
        cv2.rectangle(out, (bx0, by0 + bar_h - fill), (bx0 + bar_w, by0 + bar_h), col, -1)
        cv2.putText(out, f"{int(confidence * 100)}%", (bx0 - 10, by0 + bar_h + 15), self._font, 0.35, WHITE, 1)

        # --- Debug mode ---
        if self._debug_mode:
            out = self._draw_debug(out, debug_info, target_pos)

        return out

    def _draw_debug(
        self,
        frame: np.ndarray,
        info: Dict[str, Any],
        target_pos: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        """Render debug insets and numeric overlays."""
        h, w = frame.shape[:2]
        inset_h, inset_w = h // 4, w // 4

        # Diff image (top-left inset)
        diff_img = info.get("diff_image")
        if diff_img is not None:
            small = cv2.resize(diff_img, (inset_w, inset_h))
            if len(small.shape) == 2:
                small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            frame[0:inset_h, 0:inset_w] = small
            cv2.putText(frame, "DIFF", (5, inset_h + 15), self._font, 0.4, WHITE, 1)

        # Threshold image (below diff inset)
        thresh_img = info.get("thresh_image")
        if thresh_img is not None:
            small = cv2.resize(thresh_img, (inset_w, inset_h))
            if len(small.shape) == 2:
                small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            y0 = inset_h + 20
            if y0 + inset_h <= h:
                frame[y0 : y0 + inset_h, 0:inset_w] = small
                cv2.putText(frame, "THRESH", (5, y0 + inset_h + 15), self._font, 0.4, WHITE, 1)

        # All blobs
        blobs = info.get("all_blobs", [])
        for bx, by, _ in blobs:
            cv2.circle(frame, (int(bx), int(by)), 4, YELLOW, 1)

        # Search radius circle
        search_center = info.get("search_center")
        if search_center is not None:
            cv2.circle(
                frame,
                (int(search_center[0]), int(search_center[1])),
                info.get("search_radius", 100),
                CYAN,
                1,
                cv2.LINE_AA,
            )

        # Numeric debug text
        y_text = h - 80
        stats = [
            f"Features: {info.get('num_features', 0)}",
            f"Tracked: {info.get('num_tracked', 0)}",
            f"Inlier: {info.get('inlier_ratio', 0):.2f}",
            f"Blobs: {info.get('num_blobs', 0)}",
            f"Det: {'Y' if info.get('detection_found') else 'N'}",
        ]
        for i, line in enumerate(stats):
            cv2.putText(frame, line, (10, y_text + i * 16), self._font, 0.38, WHITE, 1)

        return frame
