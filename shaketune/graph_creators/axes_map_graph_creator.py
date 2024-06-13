# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: axes_map_graph_creator.py
# Description: Implements the axes map detection script for Shake&Tune, including
#              calibration tools and graph creation for 3D printer vibration analysis.


import optparse
import os
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pywt
from scipy import stats

matplotlib.use('Agg')

from ..helpers.common_func import parse_log
from ..helpers.console_output import ConsoleOutput
from ..shaketune_config import ShakeTuneConfig
from .graph_creator import GraphCreator

KLIPPAIN_COLORS = {
    'purple': '#70088C',
    'orange': '#FF8D32',
    'dark_purple': '#150140',
    'dark_orange': '#F24130',
    'red_pink': '#F2055C',
}
MACHINE_AXES = ['x', 'y', 'z']


class AxesMapGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config, 'axes map')
        self._accel: Optional[int] = None
        self._segment_length: Optional[float] = None

    def configure(self, accel: int, segment_length: float) -> None:
        self._accel = accel
        self._segment_length = segment_length

    def create_graph(self) -> None:
        lognames = self._move_and_prepare_files(
            glob_pattern='shaketune-axesmap_*.csv',
            min_files_required=3,
            custom_name_func=lambda f: f.stem.split('_')[1].upper(),
        )
        fig = axesmap_calibration(
            lognames=[str(path) for path in lognames],
            accel=self._accel,
            fixed_length=self._segment_length,
            st_version=self._version,
        )
        self._save_figure_and_cleanup(fig, lognames)

    def clean_old_files(self, keep_results: int = 3) -> None:
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)
        if len(files) <= keep_results:
            return  # No need to delete any files
        for old_file in files[keep_results:]:
            file_date = '_'.join(old_file.stem.split('_')[1:3])
            for suffix in {'X', 'Y', 'Z'}:
                csv_file = self._folder / f'axesmap_{file_date}_{suffix}.csv'
                csv_file.unlink(missing_ok=True)
            old_file.unlink()


######################################################################
# Computation
######################################################################


def wavelet_denoise(data: np.ndarray, wavelet: str = 'db1', level: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    coeffs = pywt.wavedec(data, wavelet, mode='smooth')
    threshold = np.median(np.abs(coeffs[-level])) / 0.6745 * np.sqrt(2 * np.log(len(data)))
    new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised_data = pywt.waverec(new_coeffs, wavelet)

    # Compute noise by subtracting denoised data from original data
    noise = data - denoised_data[: len(data)]
    return denoised_data, noise


def integrate_trapz(accel: np.ndarray, time: np.ndarray) -> np.ndarray:
    return np.array([np.trapz(accel[:i], time[:i]) for i in range(2, len(time) + 1)])


def process_acceleration_data(
    time: np.ndarray, accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray
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
    accel_x, noise_x = wavelet_denoise(accel_x)
    accel_y, noise_y = wavelet_denoise(accel_y)
    accel_z, noise_z = wavelet_denoise(accel_z)

    # Integrate acceleration to get velocity using trapezoidal rule
    velocity_x = integrate_trapz(accel_x, time)
    velocity_y = integrate_trapz(accel_y, time)
    velocity_z = integrate_trapz(accel_z, time)

    # Correct drift in velocity by resetting to zero at the beginning and end
    velocity_x -= np.linspace(velocity_x[0], velocity_x[-1], len(velocity_x))
    velocity_y -= np.linspace(velocity_y[0], velocity_y[-1], len(velocity_y))
    velocity_z -= np.linspace(velocity_z[0], velocity_z[-1], len(velocity_z))

    # Integrate velocity to get position using trapezoidal rule
    position_x = integrate_trapz(velocity_x, time[1:])
    position_y = integrate_trapz(velocity_y, time[1:])
    position_z = integrate_trapz(velocity_z, time[1:])

    noise_intensity = np.mean([np.std(noise_x), np.std(noise_y), np.std(noise_z)])

    return offset_x, offset_y, offset_z, position_x, position_y, position_z, noise_intensity


def scale_positions_to_fixed_length(
    position_x: np.ndarray, position_y: np.ndarray, position_z: np.ndarray, fixed_length: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Calculate the total distance traveled in 3D space
    total_distance = np.sqrt(np.diff(position_x) ** 2 + np.diff(position_y) ** 2 + np.diff(position_z) ** 2).sum()
    scale_factor = fixed_length / total_distance

    # Apply the scale factor to the positions
    position_x *= scale_factor
    position_y *= scale_factor
    position_z *= scale_factor

    return position_x, position_y, position_z


def find_nearest_perfect_vector(average_direction_vector: np.ndarray) -> Tuple[np.ndarray, float]:
    # Define the perfect vectors
    perfect_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Find the nearest perfect vector
    dot_products = perfect_vectors @ average_direction_vector
    nearest_vector_idx = np.argmax(dot_products)
    nearest_vector = perfect_vectors[nearest_vector_idx]

    # Calculate the angle error
    angle_error = np.arccos(dot_products[nearest_vector_idx]) * 180 / np.pi

    return nearest_vector, angle_error


def linear_regression_direction(
    position_x: np.ndarray, position_y: np.ndarray, position_z: np.ndarray, trim_length: float = 0.25
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


######################################################################
# Graphing
######################################################################


def plot_compare_frequency(
    ax: plt.Axes, time: np.ndarray, accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray, offset: float, i: int
) -> None:
    # Plot acceleration data
    ax.plot(
        time,
        accel_x,
        label='X' if i == 0 else '',
        color=KLIPPAIN_COLORS['purple'],
        linewidth=0.5,
        zorder=50 if i == 0 else 10,
    )
    ax.plot(
        time,
        accel_y,
        label='Y' if i == 0 else '',
        color=KLIPPAIN_COLORS['orange'],
        linewidth=0.5,
        zorder=50 if i == 1 else 10,
    )
    ax.plot(
        time,
        accel_z,
        label='Z' if i == 0 else '',
        color=KLIPPAIN_COLORS['red_pink'],
        linewidth=0.5,
        zorder=50 if i == 2 else 10,
    )

    # Setting axis parameters, grid and graph title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (mm/s²)')

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax.set_title(
        'Acceleration (gravity offset removed)',
        fontsize=14,
        color=KLIPPAIN_COLORS['dark_orange'],
        weight='bold',
    )

    ax.legend(loc='upper left', prop=fontP)

    # Add gravity offset to the graph
    if i == 0:
        ax2 = ax.twinx()  # To split the legends in two box
        ax2.yaxis.set_visible(False)
        ax2.plot([], [], ' ', label=f'Measured gravity: {offset / 1000:0.3f} m/s²')
        ax2.legend(loc='upper right', prop=fontP)


def plot_3d_path(
    ax: plt.Axes,
    i: int,
    position_x: np.ndarray,
    position_y: np.ndarray,
    position_z: np.ndarray,
    average_direction_vector: np.ndarray,
    angle_error: float,
) -> None:
    ax.plot(position_x, position_y, position_z, color=KLIPPAIN_COLORS['orange'], linestyle=':', linewidth=2)
    ax.scatter(position_x[0], position_y[0], position_z[0], color=KLIPPAIN_COLORS['red_pink'], zorder=10)
    ax.text(
        position_x[0] + 1,
        position_y[0],
        position_z[0],
        str(i + 1),
        color='black',
        fontsize=16,
        fontweight='bold',
        zorder=20,
    )

    # Plot the average direction vector
    start_position = np.array([position_x[0], position_y[0], position_z[0]])
    end_position = start_position + average_direction_vector * np.linalg.norm(
        [position_x[-1] - position_x[0], position_y[-1] - position_y[0], position_z[-1] - position_z[0]]
    )
    axes = ['X', 'Y', 'Z']
    ax.plot(
        [start_position[0], end_position[0]],
        [start_position[1], end_position[1]],
        [start_position[2], end_position[2]],
        label=f'{axes[i]} angle: {angle_error:0.2f}°',
        color=KLIPPAIN_COLORS['purple'],
        linestyle='-',
        linewidth=2,
    )

    # Setting axis parameters, grid and graph title
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_zlabel('Z Position (mm)')

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax.set_title(
        'Estimated movement in 3D space',
        fontsize=14,
        color=KLIPPAIN_COLORS['dark_orange'],
        weight='bold',
    )

    ax.legend(loc='upper left', prop=fontP)


def format_direction_vector(vectors: List[np.ndarray]) -> str:
    formatted_vector = []
    for vector in vectors:
        for i in range(len(vector)):
            if vector[i] > 0:
                formatted_vector.append(MACHINE_AXES[i])
                break
            elif vector[i] < 0:
                formatted_vector.append(f'-{MACHINE_AXES[i]}')
                break
    return ', '.join(formatted_vector)


######################################################################
# Startup and main routines
######################################################################


def axesmap_calibration(
    lognames: List[str], fixed_length: float, accel: Optional[float] = None, st_version: str = 'unknown'
) -> plt.Figure:
    # Parse data from the log files while ignoring CSV in the wrong format (sorted by axis name)
    raw_datas = {}
    for logname in lognames:
        data = parse_log(logname)
        if data is not None:
            _axis = logname.split('_')[-1].split('.')[0].lower()
            raw_datas[_axis] = data

    if len(raw_datas) != 3:
        raise ValueError('This tool needs 3 CSVs to work with (like axesmap_X.csv, axesmap_Y.csv and axesmap_Z.csv)')

    fig, ((ax1, ax2)) = plt.subplots(
        1,
        2,
        gridspec_kw={
            'width_ratios': [5, 3],
            'bottom': 0.080,
            'top': 0.840,
            'left': 0.055,
            'right': 0.960,
            'hspace': 0.166,
            'wspace': 0.060,
        },
    )
    fig.set_size_inches(15, 7)
    ax2.remove()
    ax2 = fig.add_subplot(122, projection='3d')

    cumulative_start_position = np.array([0, 0, 0])
    direction_vectors = []
    total_noise_intensity = 0.0
    for i, machine_axis in enumerate(MACHINE_AXES):
        if machine_axis not in raw_datas:
            raise ValueError(f'Missing CSV file for axis {machine_axis}')

        # Get the accel data according to the current axes_map
        time = raw_datas[machine_axis][:, 0]
        accel_x = raw_datas[machine_axis][:, 1]
        accel_y = raw_datas[machine_axis][:, 2]
        accel_z = raw_datas[machine_axis][:, 3]

        offset_x, offset_y, offset_z, position_x, position_y, position_z, noise_intensity = process_acceleration_data(
            time, accel_x, accel_y, accel_z
        )
        position_x, position_y, position_z = scale_positions_to_fixed_length(
            position_x, position_y, position_z, fixed_length
        )
        position_x += cumulative_start_position[0]
        position_y += cumulative_start_position[1]
        position_z += cumulative_start_position[2]

        gravity = np.linalg.norm(np.array([offset_x, offset_y, offset_z]))
        average_direction_vector = linear_regression_direction(position_x, position_y, position_z)
        direction_vector, angle_error = find_nearest_perfect_vector(average_direction_vector)
        ConsoleOutput.print(
            f'Machine axis {machine_axis.upper()} -> nearest accelerometer direction vector: {direction_vector} (angle error: {angle_error:.2f}°)'
        )
        direction_vectors.append(direction_vector)

        total_noise_intensity += noise_intensity

        plot_compare_frequency(ax1, time, accel_x, accel_y, accel_z, gravity, i)
        plot_3d_path(ax2, i, position_x, position_y, position_z, average_direction_vector, angle_error)

        # Update the cumulative start position for the next segment
        cumulative_start_position = np.array([position_x[-1], position_y[-1], position_z[-1]])

    average_noise_intensity = total_noise_intensity / len(raw_datas)
    if average_noise_intensity <= 350:
        average_noise_intensity_text = '-> OK'
    elif 350 < average_noise_intensity <= 700:
        average_noise_intensity_text = '-> WARNING: accelerometer noise is a bit high'
    else:
        average_noise_intensity_text = '-> ERROR: accelerometer noise is too high!'

    formatted_direction_vector = format_direction_vector(direction_vectors)
    ConsoleOutput.print(f'--> Detected axes_map: {formatted_direction_vector}')
    ConsoleOutput.print(
        f'Average accelerometer noise level: {average_noise_intensity:.2f} mm/s² {average_noise_intensity_text}'
    )

    # Add title
    title_line1 = 'AXES MAP CALIBRATION TOOL'
    fig.text(
        0.060, 0.947, title_line1, ha='left', va='bottom', fontsize=20, color=KLIPPAIN_COLORS['purple'], weight='bold'
    )
    try:
        filename = lognames[0].split('/')[-1]
        dt = datetime.strptime(f"{filename.split('_')[1]} {filename.split('_')[2]}", '%Y%m%d %H%M%S')
        title_line2 = dt.strftime('%x %X')
        if accel is not None:
            title_line2 += f' -- at {accel:0.0f} mm/s²'
    except Exception:
        ConsoleOutput.print(
            f'Warning: CSV filenames look to be different than expected ({lognames[0]}, {lognames[1]}, {lognames[2]})'
        )
        title_line2 = lognames[0].split('/')[-1] + ' ...'
    fig.text(0.060, 0.939, title_line2, ha='left', va='top', fontsize=16, color=KLIPPAIN_COLORS['dark_purple'])

    title_line3 = f'| Detected axes_map: {formatted_direction_vector}'
    title_line4 = f'| Accelerometer noise level: {average_noise_intensity:.2f} mm/s² {average_noise_intensity_text}'
    fig.text(0.50, 0.985, title_line3, ha='left', va='top', fontsize=14, color=KLIPPAIN_COLORS['dark_purple'])
    fig.text(0.50, 0.950, title_line4, ha='left', va='top', fontsize=11, color=KLIPPAIN_COLORS['dark_purple'])

    # Adding a small Klippain logo to the top left corner of the figure
    ax_logo = fig.add_axes([0.001, 0.894, 0.105, 0.105], anchor='NW')
    ax_logo.imshow(plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'klippain.png')))
    ax_logo.axis('off')

    # Adding Shake&Tune version in the top right corner
    if st_version != 'unknown':
        fig.text(0.995, 0.980, st_version, ha='right', va='bottom', fontsize=8, color=KLIPPAIN_COLORS['purple'])

    return fig


def main():
    # Parse command-line arguments
    usage = '%prog [options] <raw logs>'
    opts = optparse.OptionParser(usage)
    opts.add_option('-o', '--output', type='string', dest='output', default=None, help='filename of output graph')
    opts.add_option(
        '-a', '--accel', type='string', dest='accel', default=None, help='acceleration value used to do the movements'
    )
    opts.add_option(
        '-l', '--length', type='float', dest='length', default=None, help='recorded length for each segment'
    )
    options, args = opts.parse_args()
    if len(args) < 1:
        opts.error('No CSV file(s) to analyse')
    if options.accel is None:
        opts.error('You must specify the acceleration value used when generating the CSV file (option -a)')
    try:
        accel_value = float(options.accel)
    except ValueError:
        opts.error('Invalid acceleration value. It should be a numeric value.')
    if options.length is None:
        opts.error('You must specify the length of the measured segments (option -l)')
    try:
        length_value = float(options.length)
    except ValueError:
        opts.error('Invalid length value. It should be a numeric value.')
    if options.output is None:
        opts.error('You must specify an output file.png to use the script (option -o)')

    fig = axesmap_calibration(args, length_value, accel_value, 'unknown')
    fig.savefig(options.output, dpi=150)


if __name__ == '__main__':
    main()
