"""Real-time directional audio guidance via beeps, pitch, and stereo panning."""

import math
import threading
import time
from typing import Optional

import numpy as np

try:
    import pygame
    import pygame.mixer

    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False


def _generate_tone(
    frequency: float,
    duration_ms: int = 80,
    sample_rate: int = 44100,
    volume: float = 0.5,
) -> np.ndarray:
    """Create a mono sine-wave tone as a 16-bit PCM array."""
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    wave = np.sin(2 * math.pi * frequency * t)
    # Fade in/out to avoid clicks (5 ms ramp)
    ramp = min(int(sample_rate * 0.005), n_samples // 2)
    if ramp > 0:
        fade = np.linspace(0, 1, ramp)
        wave[:ramp] *= fade
        wave[-ramp:] *= fade[::-1]
    wave = (wave * volume * 32767).astype(np.int16)
    return wave


class AudioCues:
    """Generates real-time audio guidance so the pilot doesn't need the screen.

    Beep rate  → total error magnitude (faster = further from centre)
    Beep pitch → vertical error (higher = target above centre)
    Stereo pan → horizontal error (left/right)
    """

    # Pre-defined pitch levels (Hz) mapped to vertical error
    _PITCHES = [300, 400, 500, 600, 700, 800, 900]

    def __init__(self, config: dict) -> None:
        self._enabled: bool = config.get("enabled", True) and _HAS_PYGAME
        self._volume: float = config.get("volume", 0.7)
        self._min_interval: float = config.get("min_beep_interval", 0.1)
        self._max_interval: float = config.get("max_beep_interval", 1.0)
        self._deadzone: float = config.get("deadzone_pixels", 15)
        self._muted: bool = False

        self._last_beep_time: float = 0.0
        self._lock = threading.Lock()
        self._frame_half_w: float = 360.0
        self._frame_half_h: float = 240.0

        # Pre-generate tones for each pitch
        self._tones: dict = {}
        if self._enabled:
            try:
                pygame.mixer.pre_init(44100, -16, 2, 512)
                pygame.mixer.init()
                for freq in self._PITCHES:
                    mono = _generate_tone(freq, duration_ms=80, volume=self._volume)
                    # Duplicate to stereo
                    stereo = np.column_stack((mono, mono))
                    snd = pygame.sndarray.make_sound(stereo)
                    self._tones[freq] = snd
            except Exception:
                self._enabled = False

        # Warning tone (distinct from directional beeps)
        if self._enabled:
            try:
                mono = _generate_tone(200, duration_ms=300, volume=self._volume * 0.6)
                stereo = np.column_stack((mono, mono))
                self._lost_tone = pygame.sndarray.make_sound(stereo)
            except Exception:
                self._lost_tone = None
        else:
            self._lost_tone = None

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_half_w = width / 2.0
        self._frame_half_h = height / 2.0

    def toggle_mute(self) -> None:
        self._muted = not self._muted

    @property
    def is_muted(self) -> bool:
        return self._muted

    def adjust_volume(self, delta: float) -> None:
        self._volume = max(0.0, min(1.0, self._volume + delta))
        # Regenerate tones at the new volume
        if self._enabled:
            for freq in self._PITCHES:
                mono = _generate_tone(freq, duration_ms=80, volume=self._volume)
                stereo = np.column_stack((mono, mono))
                self._tones[freq] = pygame.sndarray.make_sound(stereo)

    # Zone thresholds (must stay in sync with visual_overlay.get_zone)
    _ZONE_GREEN_MAX = 0.35
    _ZONE_YELLOW_MAX = 0.65

    def update(
        self,
        error_x: Optional[float],
        error_y: Optional[float],
        confidence: float,
    ) -> None:
        """Called every frame. Decides whether and what to play.

        Beep behaviour is driven by *zone* (distance from frame centre):
          - Green  (norm <= 0.35): silence — on target.
          - Yellow (0.35 – 0.65): beep every 0.9s for course-correction cues.
          - Red    (>= 0.65):     beep every 0.4s — target near edge.
        """
        if not self._enabled or self._muted:
            return

        # Nothing to do if not tracking
        if error_x is None or error_y is None:
            return

        # Low-confidence warning
        if confidence < 0.3:
            now = time.time()
            if now - self._last_beep_time > 1.0 and self._lost_tone is not None:
                self._lost_tone.play()
                self._last_beep_time = now
            return

        # Compute normalised distance (0 = centre, 1 = edge)
        norm = 0.0
        if self._frame_half_w and self._frame_half_h:
            norm = max(
                abs(error_x) / self._frame_half_w,
                abs(error_y) / self._frame_half_h,
            )

        # Green zone — on target, stay silent
        if norm < self._ZONE_GREEN_MAX:
            return

        now = time.time()

        # Yellow zone — beep every 0.9s
        # Red zone   — beep every 0.4s
        if norm < self._ZONE_YELLOW_MAX:
            interval = 0.9
        else:
            interval = 0.4

        if now - self._last_beep_time < interval:
            return

        # Pitch: vertical error → higher pitch when target is above
        vert_norm = -error_y / self._frame_half_h  # negative y = above centre → positive
        vert_norm = max(-1.0, min(1.0, vert_norm))
        idx = int((vert_norm + 1.0) / 2.0 * (len(self._PITCHES) - 1))
        idx = max(0, min(len(self._PITCHES) - 1, idx))
        freq = self._PITCHES[idx]

        tone = self._tones.get(freq)
        if tone is None:
            return

        # Stereo panning via channel volumes
        horiz_norm = error_x / self._frame_half_w  # positive = target right
        horiz_norm = max(-1.0, min(1.0, horiz_norm))
        left_vol = max(0.0, min(1.0, 0.5 - horiz_norm * 0.5))
        right_vol = max(0.0, min(1.0, 0.5 + horiz_norm * 0.5))

        try:
            channel = tone.play()
            if channel is not None:
                channel.set_volume(left_vol, right_vol)
        except Exception:
            pass

        self._last_beep_time = now

    def shutdown(self) -> None:
        if self._enabled:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
