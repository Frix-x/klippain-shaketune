# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: axes_map_computation.py
# Description: Computation for axes map detection using velocity-based algorithm

from typing import Dict, List, Optional, Tuple

import numpy as np

from ...helpers.accelerometer import Measurement
from ...helpers.console_output import ConsoleOutput
from ..base_models import GraphMetadata
from ..computation_results import AxesMapResult

MACHINE_AXES = ['x', 'y', 'z']
ACCEL_AXES = ['x', 'y', 'z']

# Detection threshold for 2-axis machines (bed moves on one axis)
NOISE_CONFIDENCE_THRESHOLD = 0.3  # Below this + low velocity = noise-only axis


def _parse_axes_map_to_inverse_matrix(axes_map_str: Optional[str]) -> Optional[np.ndarray]:
    """Parse axes_map string and return inverse transformation matrix.

    When Klipper has an axes_map configured, it transforms the accelerometer data.
    This function computes the inverse transformation to recover the original
    accelerometer readings, allowing the detection algorithm to work correctly.

    Args:
        axes_map_str: Format "ax, ay, az" where each is x/-x/y/-y/z/-z
                      Examples: "x, y, z" (identity), "-y, x, z"

    Returns:
        3x3 inverse transformation matrix, or None if no transformation needed
        (None, identity, or invalid format)
    """
    if axes_map_str is None:
        return None

    # Normalize: strip whitespace, lowercase
    normalized = axes_map_str.strip().lower().replace(' ', '')

    # Identity check - no transformation needed
    if normalized == 'x,y,z':
        return None

    # Parse each axis component
    parts = normalized.split(',')
    if len(parts) != 3:
        return None  # Invalid format, treat as no transformation

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    forward_matrix = np.zeros((3, 3))

    for i, part in enumerate(parts):
        sign = -1.0 if part.startswith('-') else 1.0
        axis_char = part.lstrip('-')
        if axis_char not in axis_map:
            return None  # Invalid axis
        j = axis_map[axis_char]
        forward_matrix[i, j] = sign

    # For sign-permutation matrices, inverse = transpose
    # (the transpose of an orthogonal matrix is its inverse)
    return forward_matrix.T


def _orthonormalize_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Orthonormalize a 3x3 matrix using SVD to get closest proper rotation.

    The input matrix may have small deviations from orthonormality due to
    measurement noise. SVD finds the closest proper rotation matrix.
    """
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    # Ensure proper rotation (det = +1, not reflection)
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def _extract_euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract XYZ intrinsic Euler angles (roll, pitch, yaw) from rotation matrix.

    Convention: Intrinsic XYZ means rotations applied in order: X, then Y, then Z.
    This is also known as Tait-Bryan angles.

    Returns angles in degrees as (roll, pitch, yaw).
    """
    # Handle gimbal lock (pitch = +/-90 degrees)
    sy = np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)

    if sy > 1e-6:  # Not at gimbal lock
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # Gimbal lock: pitch = +/-90 degrees
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return (np.degrees(roll), np.degrees(pitch), np.degrees(yaw))


class AxesMapComputation:
    """Computation for axes map detection using velocity-based algorithm.

    This algorithm uses low-pass filtering and velocity integration to robustly
    detect accelerometer orientation, even in the presence of structural ringing.

    The approach:
    1. Low-pass filter acceleration to remove structural ringing (50+ Hz)
    2. Integrate to velocity (single integration, minimal drift)
    3. Find peak velocity - the axis with largest peak velocity is the motion axis
    4. The sign of peak velocity indicates orientation (+/-)
    """

    # Thresholds for validation
    MIN_CONFIDENCE = 0.5
    MAX_ANGLE_ERROR = 15.0  # degrees
    EXPECTED_GRAVITY = 9810  # mm/s^2
    GRAVITY_TOLERANCE = 0.20  # 20% tolerance
    FILTER_CUTOFF = 25.0  # Hz - removes 50+ Hz ringing while preserving motion

    def __init__(
        self,
        measurements: List[Measurement],
        accel: float,
        st_version: str,
        current_axes_map: Optional[str] = None,
    ):
        self.measurements = measurements
        self.accel = accel
        self.st_version = st_version
        self.current_axes_map = current_axes_map
        self._inverse_axes_map_matrix = _parse_axes_map_to_inverse_matrix(current_axes_map)

    def compute(self) -> AxesMapResult:
        """Perform axes map detection computation."""
        if len(self.measurements) != 3:
            raise ValueError(
                f'This tool needs 3 measurements (for X, Y and Z) to work. Currently, it has {len(self.measurements)} '
                f'measurements named {[meas.get("name", "unknown") for meas in self.measurements]}'
            )

        raw_datas = self._parse_measurements()

        direction_vectors = []
        actual_directions = []
        angle_errors = []
        confidences = []
        noise_levels = []
        peak_velocities_data = []
        acceleration_data = []
        velocity_data = []
        gravities = []

        for machine_axis in MACHINE_AXES:
            if machine_axis not in raw_datas:
                raise ValueError(f'Missing measurement for axis {machine_axis}')

            result = self._process_single_axis(raw_datas[machine_axis])

            direction_vectors.append(result['direction_vector'])
            actual_directions.append(result['actual_direction'])
            angle_errors.append(result['angle_error'])
            confidences.append(result['confidence'])
            noise_levels.append(result['noise_level'])
            peak_velocities_data.append(result['peak_velocities'])
            acceleration_data.append(result['accel_data'])
            velocity_data.append(result['velocity_data'])
            gravities.append(result['gravity_magnitude'])

        gravity = np.mean(gravities)
        noise_level = np.mean(noise_levels)

        # Detect 2-axis machine (one axis is noise-only, e.g., Voron Trident, Ender3)
        extrapolated_axis = self._detect_noise_only_axis(confidences, peak_velocities_data)

        if extrapolated_axis is not None:
            # Extrapolate the missing axis from the two good ones
            direction_vectors, actual_directions = self._extrapolate_missing_axis(
                direction_vectors, actual_directions, extrapolated_axis
            )
            # Set confidence to 0 for extrapolated axis (it's computed, not measured)
            confidences[extrapolated_axis] = 0.0
            angle_errors[extrapolated_axis] = 0.0  # No angle error for extrapolated

        # Build rotation matrix from actual directions and orthonormalize
        raw_rotation_matrix = np.array(actual_directions)  # rows = actual directions
        rotation_matrix = _orthonormalize_rotation_matrix(raw_rotation_matrix)
        euler_angles = _extract_euler_xyz(rotation_matrix)

        # Validate results and format output
        quality_status = self._validate_results(direction_vectors, confidences, angle_errors, noise_level, gravity)
        formatted_direction_vector = self._format_direction_vector(direction_vectors)

        # Console output
        self._print_results(
            direction_vectors,
            angle_errors,
            euler_angles,
            noise_level,
            gravity,
            formatted_direction_vector,
            quality_status,
            extrapolated_axis,
        )

        # Create metadata
        metadata = GraphMetadata(
            title='AXES MAP CALIBRATION TOOL', version=self.st_version, additional_info={'accel': self.accel}
        )

        return AxesMapResult(
            metadata=metadata,
            measurements=self.measurements,
            acceleration_data=acceleration_data,
            velocity_data=velocity_data,
            gravity=gravity,
            noise_level=noise_level,
            quality_status=quality_status,
            peak_velocities_data=peak_velocities_data,
            direction_vectors=direction_vectors,
            actual_directions=actual_directions,
            rotation_matrix=rotation_matrix,
            euler_angles=euler_angles,
            angle_errors=angle_errors,
            confidences=confidences,
            formatted_direction_vector=formatted_direction_vector,
            accel=self.accel,
            extrapolated_axis=extrapolated_axis,
        )

    def _parse_measurements(self) -> Dict[str, np.ndarray]:
        """Parse measurements into a dict keyed by axis name."""
        raw_datas = {}
        for measurement in self.measurements:
            data = np.array(measurement['samples'])
            if data is not None:
                axis = measurement['name'].split('_')[1].lower()
                raw_datas[axis] = data
        return raw_datas

    def _process_single_axis(self, data: np.ndarray) -> Dict:
        """Process acceleration data for a single machine axis movement."""
        time = data[:, 0]
        accel_x = data[:, 1].copy()
        accel_y = data[:, 2].copy()
        accel_z = data[:, 3].copy()

        # Apply inverse transformation if an axes_map was configured
        # This recovers the original accelerometer readings before Klipper's remapping
        if self._inverse_axes_map_matrix is not None:
            accel_stack = np.vstack([accel_x, accel_y, accel_z])  # 3 x N
            accel_transformed = self._inverse_axes_map_matrix @ accel_stack
            accel_x = accel_transformed[0].copy()
            accel_y = accel_transformed[1].copy()
            accel_z = accel_transformed[2].copy()

        # Estimate sample rate
        sample_rate = len(time) / (time[-1] - time[0]) if time[-1] > time[0] else 3200

        # Step 1: Remove gravity using median (robust to motion phases)
        accel_x_clean, accel_y_clean, accel_z_clean, gravity_magnitude = self._remove_gravity_robust(
            accel_x, accel_y, accel_z
        )

        # Step 2: Estimate noise level (before filtering)
        noise_level = np.mean(
            [
                self._estimate_noise_level(accel_x_clean, sample_rate),
                self._estimate_noise_level(accel_y_clean, sample_rate),
                self._estimate_noise_level(accel_z_clean, sample_rate),
            ]
        )

        # Step 3: Low-pass filter to remove structural ringing
        accel_x_filt = self._lowpass_filter(accel_x_clean, sample_rate)
        accel_y_filt = self._lowpass_filter(accel_y_clean, sample_rate)
        accel_z_filt = self._lowpass_filter(accel_z_clean, sample_rate)

        # Step 4: Integrate to velocity
        vel_x = self._integrate_to_velocity(accel_x_filt, time)
        vel_y = self._integrate_to_velocity(accel_y_filt, time)
        vel_z = self._integrate_to_velocity(accel_z_filt, time)

        # Step 5: Correct linear drift in velocity
        vel_x = self._correct_velocity_drift(vel_x)
        vel_y = self._correct_velocity_drift(vel_y)
        vel_z = self._correct_velocity_drift(vel_z)

        # Step 6: Detect direction from peak velocities
        velocities = {'x': vel_x, 'y': vel_y, 'z': vel_z}
        direction_vector, actual_direction, confidence, angle_error, peak_velocities = self._detect_direction(
            velocities
        )

        return {
            'direction_vector': direction_vector,
            'actual_direction': actual_direction,
            'confidence': confidence,
            'angle_error': angle_error,
            'noise_level': noise_level,
            'peak_velocities': peak_velocities,
            'gravity_magnitude': gravity_magnitude,
            'accel_data': (time, (accel_x_filt, accel_y_filt, accel_z_filt)),
            'velocity_data': (time, (vel_x, vel_y, vel_z)),
        }

    def _remove_gravity_robust(
        self, accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Remove gravity offset using median (robust to motion phases)."""
        gravity_x = np.median(accel_x)
        gravity_y = np.median(accel_y)
        gravity_z = np.median(accel_z)

        gravity_magnitude = np.sqrt(gravity_x**2 + gravity_y**2 + gravity_z**2)

        return (
            accel_x - gravity_x,
            accel_y - gravity_y,
            accel_z - gravity_z,
            gravity_magnitude,
        )

    def _lowpass_filter(self, data: np.ndarray, sample_rate: float) -> np.ndarray:
        """Low-pass filter using cascaded moving average.

        Two passes of moving average approximate a 2nd order Butterworth.
        Removes structural ringing (50+ Hz) while preserving motion signal.
        """
        # Window size for approximate cutoff frequency
        window_size = int(sample_rate / self.FILTER_CUTOFF / 2)
        window_size = max(3, window_size | 1)  # Ensure odd and at least 3

        kernel = np.ones(window_size) / window_size

        # Two passes for better frequency response
        filtered = np.convolve(data, kernel, mode='same')
        filtered = np.convolve(filtered, kernel, mode='same')

        return filtered

    def _integrate_to_velocity(self, accel: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Integrate acceleration to velocity using trapezoidal rule."""
        dt = np.diff(time)
        velocity = np.zeros(len(accel))
        velocity[1:] = np.cumsum((accel[:-1] + accel[1:]) / 2 * dt)
        return velocity

    def _correct_velocity_drift(self, velocity: np.ndarray) -> np.ndarray:
        """Remove linear drift from velocity signal."""
        n = len(velocity)
        if n < 2:
            return velocity

        # Remove linear trend (velocity should start and end at ~0)
        slope = (velocity[-1] - velocity[0]) / (n - 1)
        x = np.arange(n)
        velocity_corrected = velocity - (velocity[0] + slope * x)
        return velocity_corrected

    def _detect_direction(
        self, velocities: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float, float, Dict[str, float]]:
        """Determine axis and direction from peak velocities.

        Returns:
            direction_vector: Perfect unit vector for detected axis
            actual_direction: Actual normalized direction from velocity peaks
            confidence: Detection confidence 0-1
            angle_error: Angle between actual and perfect direction in degrees
            peak_velocities: Dict of peak velocity values per axis
        """
        # Find peak velocity for each axis (signed)
        peak_velocities = {}
        for axis, vel in velocities.items():
            max_vel = np.max(vel)
            min_vel = np.min(vel)
            # Use the extreme with larger magnitude
            if abs(max_vel) >= abs(min_vel):
                peak_velocities[axis] = max_vel
            else:
                peak_velocities[axis] = min_vel

        # Build actual direction vector from peak velocities
        raw_direction = np.array([peak_velocities['x'], peak_velocities['y'], peak_velocities['z']])
        direction_norm = np.linalg.norm(raw_direction)

        if direction_norm > 0:
            actual_direction = raw_direction / direction_norm
        else:
            actual_direction = raw_direction

        # Find axis with largest absolute peak velocity
        abs_peaks = {axis: abs(vel) for axis, vel in peak_velocities.items()}
        primary_axis = max(abs_peaks, key=abs_peaks.get)
        primary_sign = 1.0 if peak_velocities[primary_axis] > 0 else -1.0

        # Build perfect direction vector
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[primary_axis]
        direction_vector = np.array([0.0, 0.0, 0.0])
        direction_vector[axis_idx] = primary_sign

        # Compute angle error between actual and perfect direction
        dot_product = np.dot(actual_direction, direction_vector)
        angle_error = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

        # Compute confidence based on velocity ratio
        sorted_peaks = sorted(abs_peaks.values(), reverse=True)
        if sorted_peaks[1] > 0:
            dominance_ratio = sorted_peaks[0] / sorted_peaks[1]
        else:
            dominance_ratio = float('inf')

        confidence = min(1.0, max(0.0, (dominance_ratio - 1) / 4))

        return direction_vector, actual_direction, confidence, angle_error, peak_velocities

    def _estimate_noise_level(self, accel_axis: np.ndarray, sample_rate: float) -> float:
        """Estimate noise level using moving average residual with MAD."""
        if len(accel_axis) < 10:
            return 0.0

        # Moving average window: ~10ms
        window_size = max(5, int(sample_rate * 0.01))
        if window_size % 2 == 0:
            window_size += 1

        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(accel_axis, kernel, mode='same')

        # High-frequency noise component
        noise = accel_axis - smoothed

        # MAD-based robust noise estimation
        noise_level = 1.4826 * np.median(np.abs(noise - np.median(noise)))

        return noise_level

    def _detect_noise_only_axis(
        self,
        confidences: List[float],
        peak_velocities_data: List[Dict[str, float]],
    ) -> Optional[int]:
        """Detect if exactly one axis has noise-only data (2-axis machine).

        On machines like Voron Trident or Ender3, the accelerometer doesn't move
        on one axis (bed moves instead). This method detects that situation.

        Returns:
            Index of noise-only axis (0=X, 1=Y, 2=Z) or None if all axes have signal.
            Raises ValueError if more than one axis is noise-only.
        """
        # Get peak velocity magnitudes for each axis
        peak_mags = []
        for pvd in peak_velocities_data:
            peak_mags.append(max(abs(v) for v in pvd.values()))

        # Dynamic threshold: 1/4 of max velocity from good axes
        max_velocity = max(peak_mags)
        velocity_threshold = max_velocity / 4.0

        # Identify noise-only axes (low confidence AND low velocity)
        noise_axes = []
        for i in range(3):
            is_noise = confidences[i] < NOISE_CONFIDENCE_THRESHOLD and peak_mags[i] < velocity_threshold
            if is_noise:
                noise_axes.append(i)

        if len(noise_axes) == 0:
            return None  # All axes good - normal 3-axis machine
        elif len(noise_axes) == 1:
            return noise_axes[0]  # One axis to extrapolate - 2-axis machine
        else:
            raise ValueError(
                f'Multiple axes ({len(noise_axes)}) have no signal. '
                'Ensure accelerometer is properly mounted and moves with toolhead.'
            )

    def _extrapolate_missing_axis(
        self,
        direction_vectors: List[np.ndarray],
        actual_directions: List[np.ndarray],
        noise_axis_idx: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extrapolate missing axis using cross product of two good axes.

        When exactly one axis has no signal, we can compute its direction
        from the other two using the cross product (orthonormal constraint).
        """
        good_indices = [i for i in range(3) if i != noise_axis_idx]
        i, j = good_indices

        d1 = direction_vectors[i]
        d2 = direction_vectors[j]

        # Cross product gives the third orthogonal axis
        cross = np.cross(d1, d2)

        # Sign correction for right-handed coordinate system
        # X(0) x Y(1) = +Z(2), Y(1) x Z(2) = +X(0), Z(2) x X(0) = +Y(1)
        # With sorted indices: (0,1)->2 (+), (1,2)->0 (+), (0,2)->1 (-)
        if (i, j) == (0, 2):  # X x Z should give -Y
            cross = -cross

        cross_normalized = cross / np.linalg.norm(cross)

        new_direction_vectors = list(direction_vectors)
        new_actual_directions = list(actual_directions)
        new_direction_vectors[noise_axis_idx] = cross_normalized
        new_actual_directions[noise_axis_idx] = cross_normalized

        return new_direction_vectors, new_actual_directions

    def _validate_results(
        self,
        direction_vectors: List[np.ndarray],
        confidences: List[float],
        angle_errors: List[float],
        noise_level: float,
        gravity: float,
    ) -> Dict:
        """Validate detection results and return quality status."""
        messages = []
        status = 'ok'

        # Check 1: All axes detected uniquely
        detected_axes = []
        for dv in direction_vectors:
            axis_idx = int(np.argmax(np.abs(dv)))
            detected_axes.append(axis_idx)

        if len(set(detected_axes)) != 3:
            status = 'error'
            messages.append('Same accelerometer axis detected for multiple machine axes!')

        # Check 2: Confidence levels
        avg_confidence = np.mean(confidences)
        if avg_confidence < self.MIN_CONFIDENCE:
            if status == 'ok':
                status = 'warning'
            messages.append(f'Low detection confidence ({avg_confidence:.0%})')

        # Check 3: Angle errors
        max_angle = max(angle_errors)
        if max_angle > self.MAX_ANGLE_ERROR:
            if status == 'ok':
                status = 'warning'
            messages.append(f'High angle error detected ({max_angle:.1f} degrees)')

        # Check 4: Gravity magnitude
        expected_low = self.EXPECTED_GRAVITY * (1 - self.GRAVITY_TOLERANCE)
        expected_high = self.EXPECTED_GRAVITY * (1 + self.GRAVITY_TOLERANCE)
        if not (expected_low <= gravity <= expected_high):
            if status == 'ok':
                status = 'warning'
            messages.append(f'Unusual gravity reading ({gravity / 1000:.2f} m/s^2)')

        # Check 5: Noise level
        if self.accel and noise_level > self.accel * 0.3:
            if status == 'ok':
                status = 'warning'
            messages.append(f'High noise level ({noise_level:.0f} mm/s^2)')

        if status == 'ok':
            messages.append('Detection quality: GOOD')

        return {'status': status, 'messages': messages}

    def _format_direction_vector(self, vectors: List[np.ndarray]) -> str:
        """Format direction vectors into axes_map config string."""
        formatted = []
        axes_count = {'x': 0, 'y': 0, 'z': 0}

        for vector in vectors:
            axis_idx = int(np.argmax(np.abs(vector)))
            axis_name = ACCEL_AXES[axis_idx]
            sign = '' if vector[axis_idx] > 0 else '-'
            formatted.append(f'{sign}{axis_name}')
            axes_count[axis_name] += 1

        # Validate: each axis should appear exactly once
        for count in axes_count.values():
            if count != 1:
                return 'unable to determine correctly!'

        return ', '.join(formatted)

    def _print_results(
        self,
        direction_vectors: List[np.ndarray],
        angle_errors: List[float],
        euler_angles: Tuple[float, float, float],
        noise_level: float,
        gravity: float,
        formatted_direction_vector: str,
        quality_status: Dict,
        extrapolated_axis: Optional[int] = None,
    ) -> None:
        """Print results to console."""
        # Note about axes_map inversion if one was configured
        if self._inverse_axes_map_matrix is not None:
            ConsoleOutput.print(
                f'Note: An existing axes_map ({self.current_axes_map}) was detected and temporarily deactivated for analysis'
            )

        for i, machine_axis in enumerate(MACHINE_AXES):
            dv = direction_vectors[i]
            axis_idx = int(np.argmax(np.abs(dv)))
            accel_axis = ACCEL_AXES[axis_idx]
            sign = '+' if dv[axis_idx] > 0 else '-'

            if i == extrapolated_axis:
                ConsoleOutput.print(
                    f'Machine axis {machine_axis.upper()} -> {sign}{accel_axis} '
                    f'(VIRTUAL: no accelerometer signal on this axis)'
                )
            else:
                ConsoleOutput.print(
                    f'Machine axis {machine_axis.upper()} -> {sign}{accel_axis} '
                    f'(angle error: {angle_errors[i]:.1f} degrees)'
                )

        # Explanatory note for 2-axis machines
        if extrapolated_axis is not None:
            axis_name = MACHINE_AXES[extrapolated_axis].upper()
            ConsoleOutput.print(
                f'    Note: It looks like your machine moves the bed on another axis (here: {axis_name.upper()}) like Voron Trident, Switchwire, Ender3, etc.. '
                f"Since there's no signal on this axis, the data is calculated from the other two axes. That's why it's marked as \"virtual.\""
            )

        # Print Euler angles
        roll, pitch, yaw = euler_angles
        ConsoleOutput.print(f'Accelerometer Euler orientation: X={roll:.1f}°, Y={pitch:.1f}°, Z={yaw:.1f}°')

        # Noise status
        if noise_level <= 350:
            noise_status = 'Everything is fine'
        elif noise_level <= 700:
            noise_status = 'WARNING: noise is a bit high'
        else:
            noise_status = 'ERROR: noise is too high!'
        ConsoleOutput.print(f'Dynamic noise level: {noise_level:.0f} mm/s^2 -> {noise_status}')

        ConsoleOutput.print(f'Detected gravity: {gravity / 1000:.2f} m/s^2')

        # Quality status
        if quality_status['status'] != 'ok':
            concatenated_messages = ''
            for msg in quality_status['messages']:
                if msg != 'Detection quality: GOOD':
                    concatenated_messages += f'{msg}; '
            ConsoleOutput.print(f'==> Detected axes_map: {formatted_direction_vector}  ({concatenated_messages[:-2]})')
        else:
            ConsoleOutput.print(f'==> Detected axes_map: {formatted_direction_vector}')

        # Compare with configured axes_map and provide guidance
        if self.current_axes_map is not None:
            current_normalized = self.current_axes_map.strip().lower().replace(' ', '')
            detected_normalized = formatted_direction_vector.strip().lower().replace(' ', '')
            if current_normalized != 'x,y,z':
                if current_normalized == detected_normalized:
                    ConsoleOutput.print('    Your current axes_map configuration is already correct!')
                else:
                    ConsoleOutput.print(
                        f"    Your current axes_map doesn't match! "
                        f'Please update your configuration to {detected_normalized}.'
                    )
