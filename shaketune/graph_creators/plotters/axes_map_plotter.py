# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: axes_map_plotter.py
# Description: Plotter for axes map detection graphs using 3D orientation visualization

from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..base_models import PlotterStrategy
from ..computation_results import AxesMapResult
from ..plotting_utils import AxesConfiguration, PlottingConstants

MACHINE_AXES = ['X', 'Y', 'Z']
ACCEL_AXES = ['x', 'y', 'z']
ACCEL_COLORS = {
    'x': PlottingConstants.KLIPPAIN_COLORS['purple'],
    'y': PlottingConstants.KLIPPAIN_COLORS['orange'],
    'z': PlottingConstants.KLIPPAIN_COLORS['red_pink'],
}


class AxesMapPlotter(PlotterStrategy):
    """Plotter for axes map detection graphs using 3D orientation visualization"""

    def plot(self, result: AxesMapResult) -> Figure:
        """Create axes map detection graph with 3D orientation cube and velocity sequence"""
        data = result.get_plot_data()

        # Create figure with 1x2 layout: 3D cube (1/3) + velocity sequence (2/3)
        fig = plt.figure(figsize=(15, 7))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[5, 3],
            bottom=0.080,
            top=0.840,
            left=0.055,
            right=0.960,
            wspace=0.060,
        )

        ax_velocity = fig.add_subplot(gs[0])
        ax_3d = fig.add_subplot(gs[1], projection='3d')

        # Add titles, logo, and version
        self._add_titles(fig, data)
        self.add_logo(fig)
        self.add_version_text(fig, data['st_version'])

        # Plot components
        self._plot_3d_orientation(ax_3d, data)
        self._plot_velocity_sequence(ax_velocity, data)

        return fig

    def _add_titles(self, fig: Figure, data: Dict[str, Any]) -> None:
        """Add title lines including mapping info and quality status"""
        # Parse timestamp from filename
        try:
            filename = data['measurements'][0]['name']
            dt = datetime.strptime(f'{filename.split("_")[2]} {filename.split("_")[3]}', '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X')
            if data['accel'] is not None:
                title_line2 += f' -- at {data["accel"]:0.0f} mm/s²'
        except Exception:
            title_line2 = data['measurements'][0]['name'] + ' ...'

        # Build mapping details string: "X → -z (2.3°)  Y → y (1.5°)  Z → x (extrapolated)"
        extrapolated_axis = data.get('extrapolated_axis')
        mapping_parts = []
        for i, machine_axis in enumerate(MACHINE_AXES):
            dv = data['direction_vectors'][i]
            axis_idx = int(np.argmax(np.abs(dv)))
            accel_axis = ACCEL_AXES[axis_idx]
            sign = '' if dv[axis_idx] > 0 else '-'
            if i == extrapolated_axis:
                mapping_parts.append(f'{machine_axis} → {sign}{accel_axis.upper()} (extrapolated)')
            else:
                angle = data['angle_errors'][i]
                mapping_parts.append(f'{machine_axis} → {sign}{accel_axis.upper()} (angle error: {angle:.1f}°)')
        mapping_text = '   '.join(mapping_parts)

        # Format Euler angles
        roll, pitch, yaw = data['euler_angles']
        euler_text = f'Accelerometer Euler orientation: X={roll:.1f}°  Y={pitch:.1f}°  Z={yaw:.1f}°'

        title_lines = [
            {
                'x': 0.060,
                'y': 0.947,
                'text': 'AXES MAP CALIBRATION TOOL',
                'fontsize': 20,
                'color': PlottingConstants.KLIPPAIN_COLORS['purple'],
                'weight': 'bold',
            },
            {'x': 0.060, 'y': 0.939, 'va': 'top', 'text': title_line2},
            {
                'x': 0.50,
                'y': 0.985,
                'va': 'top',
                'text': f'| Detected axes map: {data["formatted_direction_vector"]}',
                'weight': 'bold',
            },
            {
                'x': 0.501,
                'y': 0.944,
                'va': 'top',
                'fontsize': 11,
                'text': f'| {mapping_text}',
            },
            {
                'x': 0.501,
                'y': 0.910,
                'va': 'top',
                'fontsize': 11,
                'text': f'| {euler_text}',
            },
        ]
        self.add_title(fig, title_lines)

    def _plot_3d_orientation(self, ax, data: Dict[str, Any]) -> None:
        """Plot 3D orientation showing actual measured accelerometer axes relative to machine axes"""
        # Find which accelerometer axis corresponds to the extrapolated machine axis
        extrapolated_accel_idx = None
        extrapolated_axis = data.get('extrapolated_axis')
        if extrapolated_axis is not None:
            dv = data['direction_vectors'][extrapolated_axis]
            extrapolated_accel_idx = int(np.argmax(np.abs(dv)))

        # Draw machine reference axes (gray dashed)
        for i, label in enumerate(MACHINE_AXES):
            axis_vec = np.zeros(3)
            axis_vec[i] = 1.0
            ax.quiver(
                0,
                0,
                0,
                axis_vec[0] * 1.15,
                axis_vec[1] * 1.15,
                axis_vec[2] * 1.15,
                color=PlottingConstants.KLIPPAIN_COLORS['dark_purple'],
                linestyle='--',
                alpha=0.4,
                linewidth=3,
                arrow_length_ratio=0.2,
            )
            ax.text(
                axis_vec[0] * 1.28,
                axis_vec[1] * 1.28,
                axis_vec[2] * 1.28,
                label,
                fontsize=14,
                fontweight='bold',
                alpha=0.5,
                color=PlottingConstants.KLIPPAIN_COLORS['dark_purple'],
                ha='center',
                va='center',
            )

        # Use orthonormalized rotation matrix for actual measured accelerometer orientation
        # Column i of rotation_matrix = where accel axis i points in machine frame
        rotation_matrix = data['rotation_matrix']

        for i, accel_label in enumerate(ACCEL_AXES):
            accel_direction = rotation_matrix[:, i]  # Column i (orthonormal after SVD)
            color = ACCEL_COLORS[accel_label]
            is_extrapolated = i == extrapolated_accel_idx

            ax.quiver(
                0,
                0,
                0,
                accel_direction[0] * 0.9,
                accel_direction[1] * 0.9,
                accel_direction[2] * 0.9,
                color=color,
                linewidth=3,
                arrow_length_ratio=0.12,
            )

            # Add "(virtual)" suffix for extrapolated axis label
            label_text = f'{accel_label} (virtual)' if is_extrapolated else accel_label
            ax.text(
                accel_direction[0] * 1.05,
                accel_direction[1] * 1.05,
                accel_direction[2] * 1.05,
                label_text,
                fontsize=20 if not is_extrapolated else 14,
                fontweight='bold',
                color=color,
                ha='center',
                va='center',
            )

        # Configure 3D axes appearance
        AxesConfiguration.configure_axes(
            ax,
            xlabel='Machine X',
            ylabel='Machine Y',
            zlabel='Machine Z',
            title='Accelerometer Orientation',
        )
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    def _plot_velocity_sequence(self, ax, data: Dict[str, Any]) -> None:
        """Plot velocity sequence with watermark zones and confidence percentages"""
        # Concatenate velocity data from all measurements into continuous sequence
        all_times: List[float] = []
        all_vels = {'x': [], 'y': [], 'z': []}
        zone_boundaries = [0.0]
        zone_info = []

        for i, (time_data, (vel_x, vel_y, vel_z)) in enumerate(data['velocity_data']):
            time_normalized = time_data - time_data[0]
            offset = zone_boundaries[-1]
            time_shifted = time_normalized + offset

            all_times.extend(time_shifted)
            all_vels['x'].extend(vel_x)
            all_vels['y'].extend(vel_y)
            all_vels['z'].extend(vel_z)

            # Find detected axis and peak for this zone
            direction_vec = data['direction_vectors'][i]
            axis_idx = int(np.argmax(np.abs(direction_vec)))
            detected_axis = ACCEL_AXES[axis_idx]
            peak_vels = data['peak_velocities_data'][i]
            peak_val = peak_vels[detected_axis]
            confidence = data['confidences'][i]

            # Find peak position in time
            vel_array = [vel_x, vel_y, vel_z][axis_idx]
            peak_idx = np.argmax(vel_array) if peak_val > 0 else np.argmin(vel_array)
            peak_time = time_shifted[peak_idx]

            zone_info.append(
                {
                    'axis': detected_axis,
                    'peak_time': peak_time,
                    'peak_value': peak_val,
                    'confidence': confidence,
                    'zone_start': offset,
                    'zone_end': time_shifted[-1],
                }
            )
            zone_boundaries.append(time_shifted[-1])

        all_times_arr = np.array(all_times)

        # Calculate y-axis range for positioning
        all_values = list(all_vels['x']) + list(all_vels['y']) + list(all_vels['z'])
        y_min, y_max = min(all_values), max(all_values)
        y_center = (y_min + y_max) / 2
        y_range = y_max - y_min

        # Draw watermarks with confidence in background
        extrapolated_axis = data.get('extrapolated_axis')
        for i, label in enumerate(MACHINE_AXES):
            zone_center = (zone_boundaries[i] + zone_boundaries[i + 1]) / 2
            confidence = zone_info[i]['confidence']
            is_extrapolated = i == extrapolated_axis

            if is_extrapolated:
                # Gray shaded background for extrapolated zone
                ax.axvspan(
                    zone_boundaries[i],
                    zone_boundaries[i + 1],
                    alpha=0.15,
                    color='gray',
                    zorder=0,
                )
                # Gray watermark for extrapolated zone
                ax.text(
                    zone_center,
                    y_center,
                    label,
                    fontsize=55,
                    alpha=0.25,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='gray',
                    zorder=1,
                )
                # "Extrapolated" label (usually the bed axis) instead of confidence
                ax.text(
                    zone_center,
                    y_center - 15,
                    'Extrapolated\n(no signal on this axis)',
                    fontsize=9,
                    alpha=0.4,
                    ha='center',
                    va='center',
                    color='gray',
                    zorder=1,
                )
            else:
                # Normal styling for measured zones
                # Large watermark letter
                ax.text(
                    zone_center,
                    y_center,
                    label,
                    fontsize=55,
                    alpha=0.4,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color=PlottingConstants.KLIPPAIN_COLORS['dark_purple'],
                    zorder=1,
                )
                # Confidence percentage below
                ax.text(
                    zone_center,
                    y_center - 15,
                    f'Confidence: {confidence:.0%}',
                    fontsize=11,
                    alpha=0.4,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color=PlottingConstants.KLIPPAIN_COLORS['dark_purple'],
                    zorder=1,
                )

        # Draw zone separators
        for boundary in zone_boundaries[1:-1]:
            ax.axvline(
                x=boundary,
                color=PlottingConstants.KLIPPAIN_COLORS['dark_purple'],
                linestyle='--',
                alpha=0.4,
                linewidth=2,
                zorder=2,
            )

        # Draw accelerometer velocity curves
        for axis_name in ACCEL_AXES:
            ax.plot(
                all_times_arr,
                all_vels[axis_name],
                color=ACCEL_COLORS[axis_name],
                linewidth=1.2,
                label=f'Accelerometer {axis_name.upper()}',
                zorder=5,
            )

        # Mark peaks in each zone
        for info in zone_info:
            color = ACCEL_COLORS[info['axis']]
            ax.scatter(
                info['peak_time'], info['peak_value'], color=color, s=50, zorder=10, edgecolors='black', linewidths=0.5
            )
            y_offset = 8 if info['peak_value'] > 0 else -12
            ax.annotate(
                f'{info["peak_value"]:.0f} mm/s',
                (info['peak_time'], info['peak_value']),
                textcoords='offset points',
                xytext=(0, y_offset),
                fontsize=10,
                fontweight='bold',
                ha='center',
                color=color,
            )

        # Axis styling
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax.set_xlim([0, zone_boundaries[-1]])
        ax.set_ylim([y_min - y_range * 0.15, y_max + y_range * 0.15])

        # Add secondary legend with gravity and noise info
        noise = data['noise_level']
        noise_status = 'OK' if noise <= 350 else ('High' if noise <= 700 else 'Too high!')

        ax_secondary = ax.twinx()
        ax_secondary.yaxis.set_visible(False)
        ax_secondary.plot([], [], ' ', label=f'Measured gravity: {data["gravity"] / 1000:.3f} m/s²')
        ax_secondary.plot([], [], ' ', label=f'Noise: {noise:.0f} mm/s² ({noise_status})')

        fontP = AxesConfiguration.configure_axes(
            ax,
            xlabel='Time (s)',
            ylabel='Velocity (mm/s)',
            title='Movement sequence',
            legend=True,
        )
        ax_secondary.legend(loc='upper right', prop=fontP)
