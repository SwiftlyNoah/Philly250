"""Constant-velocity Kalman filter for target position smoothing."""

from typing import Tuple

import cv2
import numpy as np


class TargetKalmanFilter:
    """State = [x, y, vx, vy], measurement = [x, y].

    Shared by both the pixel tracker and YOLO tracker backends.
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1) -> None:
        self._kf = cv2.KalmanFilter(4, 2)

        # Transition matrix (constant velocity model)
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32,
        )

        # Measurement matrix — we observe x, y directly
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32,
        )

        # Process noise covariance
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Measurement noise covariance
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Error covariance (initial)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)

        self._initialized = False

    def initialize(self, x: float, y: float) -> None:
        """Set the initial state with zero velocity."""
        self._kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = True

    def predict(self) -> Tuple[float, float]:
        """Run the prediction step and return the predicted (x, y)."""
        pred = self._kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, measured_x: float, measured_y: float) -> Tuple[float, float]:
        """Fuse a measurement with the prediction; return the corrected (x, y)."""
        measurement = np.array([[measured_x], [measured_y]], dtype=np.float32)
        corrected = self._kf.correct(measurement)
        return float(corrected[0, 0]), float(corrected[1, 0])

    def get_velocity(self) -> Tuple[float, float]:
        """Return the current estimated velocity (vx, vy)."""
        state = self._kf.statePost
        return float(state[2, 0]), float(state[3, 0])

    def get_predicted_position(self, n_frames_ahead: int) -> Tuple[float, float]:
        """Extrapolate position *n_frames_ahead* into the future."""
        state = self._kf.statePost
        x = float(state[0, 0]) + float(state[2, 0]) * n_frames_ahead
        y = float(state[1, 0]) + float(state[3, 0]) * n_frames_ahead
        return x, y

    @property
    def is_initialized(self) -> bool:
        return self._initialized
