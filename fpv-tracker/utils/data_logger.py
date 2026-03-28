"""CSV data logger for post-session analysis."""

import csv
import os
import time
from typing import Optional


class DataLogger:
    """Appends one row per frame to a timestamped CSV file."""

    HEADER = [
        "frame",
        "timestamp",
        "target_x",
        "target_y",
        "error_x",
        "error_y",
        "confidence",
        "tracker_type",
        "bg_features",
        "inlier_ratio",
        "fps",
        "notes",
    ]

    def __init__(self, output_dir: str = "logs/") -> None:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._path = os.path.join(output_dir, filename)
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def log(
        self,
        frame_number: int,
        timestamp: float,
        target_x: Optional[float],
        target_y: Optional[float],
        error_x: Optional[float],
        error_y: Optional[float],
        confidence: float,
        tracker_type: str,
        num_features: int = 0,
        inlier_ratio: float = 0.0,
        fps: float = 0.0,
        notes: str = "",
    ) -> None:
        self._writer.writerow([
            frame_number,
            f"{timestamp:.6f}",
            f"{target_x:.1f}" if target_x is not None else "",
            f"{target_y:.1f}" if target_y is not None else "",
            f"{error_x:.1f}" if error_x is not None else "",
            f"{error_y:.1f}" if error_y is not None else "",
            f"{confidence:.3f}",
            tracker_type,
            num_features,
            f"{inlier_ratio:.3f}",
            f"{fps:.1f}",
            notes,
        ])

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def path(self) -> str:
        return self._path
