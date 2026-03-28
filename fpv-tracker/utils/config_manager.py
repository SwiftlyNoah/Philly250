"""Configuration manager with YAML loading, runtime updates, and hot-reload."""

import os
import threading
import time
from typing import Any, Dict, Optional

import yaml


_DEFAULTS: Dict[str, Any] = {
    "video": {
        "device_index": 0,
        "width": 720,
        "height": 480,
        "crop": [20, 20, 700, 460],
        "flip": False,
    },
    "pixel_tracker": {
        "feature_count": 300,
        "feature_quality": 0.01,
        "feature_min_distance": 10,
        "lk_window_size": 21,
        "lk_pyramid_levels": 3,
        "ransac_threshold": 3.0,
        "min_inlier_ratio": 0.5,
        "diff_threshold": 25,
        "blur_kernel": 5,
        "morph_kernel": 3,
        "min_blob_area": 1,
        "max_blob_area": 500,
        "search_radius": 100,
        "edge_margin": 40,
        "target_exclusion_radius": 50,
    },
    "kalman": {
        "process_noise": 0.01,
        "measurement_noise": 0.1,
        "max_frames_lost": 30,
        "confidence_engage": 0.7,
        "confidence_disengage": 0.2,
    },
    "yolo": {
        "model_path": "models/best.pt",
        "confidence_threshold": 0.5,
        "imgsz": 640,
    },
    "audio": {
        "enabled": True,
        "volume": 0.7,
        "min_beep_interval": 0.1,
        "max_beep_interval": 1.0,
        "deadzone_pixels": 15,
    },
    "logging": {
        "enabled": True,
        "output_dir": "logs/",
    },
    "overlay": {
        "trajectory_length": 100,
        "arrow_scale": 1.5,
        "lead_indicator_frames": 10,
    },
}


class ConfigManager:
    """Load, access, update, and optionally hot-reload a YAML config file."""

    def __init__(self, config_path: str) -> None:
        self._path = config_path
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._watch_thread: Optional[threading.Thread] = None
        self._watching = False
        self._last_mtime: float = 0.0

        self._load()

    def _load(self) -> None:
        """Load the YAML file and merge with defaults."""
        data: Dict[str, Any] = {}
        if os.path.exists(self._path):
            with open(self._path, "r") as fh:
                raw = yaml.safe_load(fh)
                if isinstance(raw, dict):
                    data = raw
            self._last_mtime = os.path.getmtime(self._path)

        merged: Dict[str, Any] = {}
        for section, defaults in _DEFAULTS.items():
            merged[section] = dict(defaults)
            if section in data and isinstance(data[section], dict):
                merged[section].update(data[section])

        # Preserve any extra top-level sections from the file
        for section in data:
            if section not in merged:
                merged[section] = data[section]

        with self._lock:
            self._data = merged

    def get(self, section: str, key: str) -> Any:
        """Retrieve a config value.  Example: ``config.get("pid", "kp_yaw")``."""
        with self._lock:
            sec = self._data.get(section, {})
            if not isinstance(sec, dict):
                return sec
            return sec.get(key)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return a *copy* of an entire config section."""
        with self._lock:
            sec = self._data.get(section, {})
            if isinstance(sec, dict):
                return dict(sec)
            return {}

    def set(self, section: str, key: str, value: Any) -> None:
        """Update a single value at runtime (e.g. from the tuning GUI)."""
        with self._lock:
            if section not in self._data:
                self._data[section] = {}
            self._data[section][key] = value

    def save(self) -> None:
        """Persist the current config back to the YAML file."""
        with self._lock:
            snapshot = {s: dict(v) if isinstance(v, dict) else v for s, v in self._data.items()}
        with open(self._path, "w") as fh:
            yaml.dump(snapshot, fh, default_flow_style=False, sort_keys=False)
        self._last_mtime = os.path.getmtime(self._path)

    # ------------------------------------------------------------------
    # Hot-reload watcher
    # ------------------------------------------------------------------
    def watch(self) -> None:
        """Start a background thread that reloads the config when the file changes."""
        if self._watching:
            return
        self._watching = True
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()

    def stop_watch(self) -> None:
        """Stop the background watcher thread."""
        self._watching = False

    def _watch_loop(self) -> None:
        while self._watching:
            time.sleep(1.0)
            if not os.path.exists(self._path):
                continue
            mtime = os.path.getmtime(self._path)
            if mtime > self._last_mtime:
                self._load()
