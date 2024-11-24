# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: axes_map_graph_creator.py
# Description: Implements the axes map detection script for Shake&Tune, including
#              calibration tools and graph creation for 3D printer vibration analysis.

from typing import List, Optional, Tuple

import numpy as np
import pywt
from scipy import stats

from ..helpers.accelerometer import Measurement, MeasurementsManager
from ..helpers.console_output import ConsoleOutput
from ..shaketune_config import ShakeTuneConfig
from .graph_creator import GraphCreator

MACHINE_AXES = ['x', 'y', 'z']


@GraphCreator.register('axes map')
class AxesMapGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config)
        self._accel: Optional[int] = None
        self._segment_length: Optional[float] = None

    def configure(self, accel: int, segment_length: float) -> None:
        self._accel = accel
        self._segment_length = segment_length

    def create_graph(self, measurements_manager: MeasurementsManager) -> None:
        computer = AxesMapComputation(
            measurements=measurements_manager.get_measurements(),
            accel=self._accel,
            fixed_length=self._segment_length,
            st_version=self._version,
        )
        computation = computer.compute()
        fig = self._plotter.plot_axes_map_detection_graph(computation)
        self._save_figure(fig, measurements_manager)


######################################################################
# Computation
######################################################################


class AxesMapComputation:
    def __init__(
        self,
        measurements: List[Measurement],
        accel: float,
        fixed_length: float,
        st_version: str,
    ):
        self.measurements = measurements
        self.accel = accel
        self.fixed_length = fixed_length
        self.st_version = st_version

    def compute(self):
        if len(self.measurements) != 3:
            raise ValueError('This tool needs 3 measurements to work with (like axesmap_X, axesmap_Y and axesmap_Z)')

        raw_datas = {}
        for measurement in self.measurements:
            data = np.array(measurement['samples'])
            if data is not None:
                _axis = measurement['name'].split('_')[1].lower()
                raw_datas[_axis] = data

        cumulative_start_position = np.array([0, 0, 0])
        direction_vectors = []
        angle_errors = []
        total_noise_intensity = 0.0
        acceleration_data = []
        position_data = []
        gravities = []
        for _, machine_axis in enumerate(MACHINE_AXES):
            if machine_axis not in raw_datas:
                raise ValueError(f'Missing measurement for axis {machine_axis}')

            # Get the accel data according to the current axes_map
            time = raw_datas[machine_axis][:, 0]
            accel_x = raw_datas[machine_axis][:, 1]
            accel_y = raw_datas[machine_axis][:, 2]
            accel_z = raw_datas[machine_axis][:, 3]

            offset_x, offset_y, offset_z, position_x, position_y, position_z, noise_intensity = (
                self._process_acceleration_data(time, accel_x, accel_y, accel_z)
            )
            position_x, position_y, position_z = self._scale_positions_to_fixed_length(
                position_x, position_y, position_z, self.fixed_length
            )
            position_x += cumulative_start_position[0]
            position_y += cumulative_start_position[1]
            position_z += cumulative_start_position[2]

            gravity = np.linalg.norm(np.array([offset_x, offset_y, offset_z]))
            average_direction_vector = self._linear_regression_direction(position_x, position_y, position_z)
            direction_vector, angle_error = self._find_nearest_perfect_vector(average_direction_vector)
            ConsoleOutput.print(
                f'Machine axis {machine_axis.upper()} -> nearest accelerometer direction vector: {direction_vector} (angle error: {angle_error:.2f}°)'
            )
            direction_vectors.append(direction_vector)
            angle_errors.append(angle_error)

            total_noise_intensity += noise_intensity

            acceleration_data.append((time, (accel_x, accel_y, accel_z)))
            position_data.append((position_x, position_y, position_z))
            gravities.append(gravity)

            # Update the cumulative start position for the next segment
            cumulative_start_position = np.array([position_x[-1], position_y[-1], position_z[-1]])

        gravity = np.mean(gravities)

        average_noise_intensity = total_noise_intensity / len(raw_datas)
        if average_noise_intensity <= 350:
            average_noise_intensity_text = '-> OK'
        elif 350 < average_noise_intensity <= 700:
            average_noise_intensity_text = '-> WARNING: accelerometer noise is a bit high'
        else:
            average_noise_intensity_text = '-> ERROR: accelerometer noise is too high!'

        average_noise_intensity_label = (
            f'Dynamic noise level: {average_noise_intensity:.2f} mm/s² {average_noise_intensity_text}'
        )
        ConsoleOutput.print(average_noise_intensity_label)

        ConsoleOutput.print(f'--> Detected gravity: {gravity / 1000 :.2f} m/s²')

        formatted_direction_vector = self._format_direction_vector(direction_vectors)
        ConsoleOutput.print(f'--> Detected axes_map: {formatted_direction_vector}')

        return {
            'acceleration_data_0': [d[0] for d in acceleration_data],
            'acceleration_data_1': [d[1] for d in acceleration_data],
            'gravity': gravity,
            'average_noise_intensity_label': average_noise_intensity_label,
            'position_data': position_data,
            'direction_vectors': direction_vectors,
            'angle_errors': angle_errors,
            'formatted_direction_vector': formatted_direction_vector,
            'measurements': self.measurements,
            'accel': self.accel,
            'st_version': self.st_version,
        }

    def _wavelet_denoise(self, data: np.ndarray, wavelet: str = 'db1', level: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        coeffs = pywt.wavedec(data, wavelet, mode='smooth')
        threshold = np.median(np.abs(coeffs[-level])) / 0.6745 * np.sqrt(2 * np.log(len(data)))
        new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        denoised_data = pywt.waverec(new_coeffs, wavelet)

        # Compute noise by subtracting denoised data from original data
        noise = data - denoised_data[: len(data)]
        return denoised_data, noise

    def _integrate_trapz(self, accel: np.ndarray, time: np.ndarray) -> np.ndarray:
        return np.array([np.trapz(accel[:i], time[:i]) for i in range(2, len(time) + 1)])

    def _process_acceleration_data(
        self, time: np.ndarray, accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, float]:
        # Calculate the constant offset (gravity component)
        offset_x = np.mean(accel_x)
        offset_y = np.mean(accel_y)
        offset_z = np.mean(accel_z)

        # Remove the constant offset from acceleration data
        accel_x -= offset_x
        accel_y -= offset_y
        accel_z -= offset_z

        # Apply wavelet denoising
        accel_x, noise_x = self._wavelet_denoise(accel_x)
        accel_y, noise_y = self._wavelet_denoise(accel_y)
        accel_z, noise_z = self._wavelet_denoise(accel_z)

        # Integrate acceleration to get velocity using trapezoidal rule
        velocity_x = self._integrate_trapz(accel_x, time)
        velocity_y = self._integrate_trapz(accel_y, time)
        velocity_z = self._integrate_trapz(accel_z, time)

        # Correct drift in velocity by resetting to zero at the beginning and end
        velocity_x -= np.linspace(velocity_x[0], velocity_x[-1], len(velocity_x))
        velocity_y -= np.linspace(velocity_y[0], velocity_y[-1], len(velocity_y))
        velocity_z -= np.linspace(velocity_z[0], velocity_z[-1], len(velocity_z))

        # Integrate velocity to get position using trapezoidal rule
        position_x = self._integrate_trapz(velocity_x, time[1:])
        position_y = self._integrate_trapz(velocity_y, time[1:])
        position_z = self._integrate_trapz(velocity_z, time[1:])

        noise_intensity = np.mean([np.std(noise_x), np.std(noise_y), np.std(noise_z)])

        return offset_x, offset_y, offset_z, position_x, position_y, position_z, noise_intensity

    def _scale_positions_to_fixed_length(
        self, position_x: np.ndarray, position_y: np.ndarray, position_z: np.ndarray, fixed_length: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Calculate the total distance traveled in 3D space
        total_distance = np.sqrt(np.diff(position_x) ** 2 + np.diff(position_y) ** 2 + np.diff(position_z) ** 2).sum()
        scale_factor = fixed_length / total_distance

        # Apply the scale factor to the positions
        position_x *= scale_factor
        position_y *= scale_factor
        position_z *= scale_factor

        return position_x, position_y, position_z

    def _find_nearest_perfect_vector(self, average_direction_vector: np.ndarray) -> Tuple[np.ndarray, float]:
        # Define the perfect vectors
        perfect_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # Find the nearest perfect vector
        dot_products = perfect_vectors @ average_direction_vector
        nearest_vector_idx = np.argmax(dot_products)
        nearest_vector = perfect_vectors[nearest_vector_idx]

        # Calculate the angle error
        angle_error = np.arccos(dot_products[nearest_vector_idx]) * 180 / np.pi

        return nearest_vector, angle_error

    def _linear_regression_direction(
        self, position_x: np.ndarray, position_y: np.ndarray, position_z: np.ndarray, trim_length: float = 0.25
    ) -> np.ndarray:
        # Trim the start and end of the position data to keep only the center of the segment
        # as the start and stop positions are not always perfectly aligned and can be a bit noisy
        t = len(position_x)
        trim_start = int(t * trim_length)
        trim_end = int(t * (1 - trim_length))
        position_x = position_x[trim_start:trim_end]
        position_y = position_y[trim_start:trim_end]
        position_z = position_z[trim_start:trim_end]

        # Compute the direction vector using linear regression over the position data
        time = np.arange(len(position_x))
        slope_x, intercept_x, _, _, _ = stats.linregress(time, position_x)
        slope_y, intercept_y, _, _, _ = stats.linregress(time, position_y)
        slope_z, intercept_z, _, _, _ = stats.linregress(time, position_z)
        end_position = np.array(
            [slope_x * time[-1] + intercept_x, slope_y * time[-1] + intercept_y, slope_z * time[-1] + intercept_z]
        )
        direction_vector = end_position - np.array([intercept_x, intercept_y, intercept_z])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        return direction_vector

    def _format_direction_vector(self, vectors: List[np.ndarray]) -> str:
        formatted_vector = []
        axes_count = {'x': 0, 'y': 0, 'z': 0}

        for vector in vectors:
            for i in range(len(vector)):
                if vector[i] > 0:
                    formatted_vector.append(MACHINE_AXES[i])
                    axes_count[MACHINE_AXES[i]] += 1
                    break
                elif vector[i] < 0:
                    formatted_vector.append(f'-{MACHINE_AXES[i]}')
                    axes_count[MACHINE_AXES[i]] += 1
                    break

        # If all axes are present, return the formatted vector
        return next(
            ('unable to determine it correctly!' for count in axes_count.values() if count != 1),
            ', '.join(formatted_vector),
        )
