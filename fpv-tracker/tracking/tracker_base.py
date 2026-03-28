"""Abstract base class that all tracker backends must implement."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class TrackerBase(ABC):
    """Interface for swappable tracker backends (pixel, YOLO, hybrid)."""

    def __init__(self) -> None:
        self._target_pos: Optional[Tuple[float, float]] = None
        self._confidence: float = 0.0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def initialize(
        self, frame: np.ndarray, target_x: Optional[float] = None, target_y: Optional[float] = None
    ) -> None:
        """Lock onto a target at the given position (defaults to frame center)."""

    @abstractmethod
    def update(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], float, Dict[str, Any]]:
        """Process one frame.

        Returns
        -------
        target_pos : (x, y) or None
        error : (error_x, error_y) relative to frame center, or None
        confidence : float 0-1
        debug_info : dict with backend-specific diagnostics
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all state and set the tracker to uninitialized."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def target_pos(self) -> Optional[Tuple[float, float]]:
        return self._target_pos

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a short identifier such as ``"pixel"`` or ``"yolo"``."""
