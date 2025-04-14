# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2022 - 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: plotter.py
# Description: Contains the Plotter class to handle the plotting logic in order to
#              create the .png graphs in Shake&Tune with all the infos using matplotlib.


import os
from datetime import datetime

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments


class Plotter:
    KLIPPAIN_COLORS = {
        'purple': '#70088C',
        'orange': '#FF8D32',
        'dark_purple': '#150140',
        'dark_orange': '#F24130',
        'red_pink': '#F2055C',
    }

    # static_frequency tool
    SPECTROGRAM_LOW_PERCENTILE_FILTER = 5

    # belts tool
    ALPHABET = 'αβγδεζηθικλμνξοπρστυφχψω'  # For paired peak names (using the Greek alphabet to avoid confusion with belt names)

    # input shaper tool
    SPECTROGRAM_LOW_PERCENTILE_FILTER = 5
    MAX_VIBRATIONS_PLOTTED = 80.0
    MAX_VIBRATIONS_PLOTTED_ZOOM = 1.25  # 1.25x max vibs values from the standard filters selection

    def __init__(self):
        # Preload logo image during Plotter initialization
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, 'klippain.png')
        self.logo_image = plt.imread(image_path)

    def add_logo(self, fig, position=None):  # noqa: B006
        if position is None:
            position = [0.001, 0.894, 0.105, 0.105]
        ax_logo = fig.add_axes(position, anchor='NW')
        ax_logo.imshow(self.logo_image)
        ax_logo.axis('off')

    def add_version_text(self, fig, st_version, position=(0.995, 0.980)):
        if st_version != 'unknown':
            fig.text(
                position[0],
                position[1],
                st_version,
                ha='right',
                va='bottom',
                fontsize=8,
                color=self.KLIPPAIN_COLORS['purple'],
            )

    def add_title(self, fig, title_lines):
        for line in title_lines:
            fig.text(
                line['x'],
                line['y'],
                line['text'],
                ha=line.get('ha', 'left'),
                va=line.get('va', 'bottom'),
                fontsize=line.get('fontsize', 16),
                color=line.get('color', self.KLIPPAIN_COLORS['dark_purple']),
                weight=line.get('weight', 'normal'),
            )

    def configure_axes(self, ax, xlabel='', ylabel='', zlabel='', title='', grid=True, sci_axes='', legend=False):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('x-small')
        if zlabel != '':
            ax.set_zlabel(zlabel)
        if title != '':
            ax.set_title(title, fontsize=14, color=self.KLIPPAIN_COLORS['dark_orange'], weight='bold')
        if grid:
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.grid(which='major', color='grey')
            ax.grid(which='minor', color='lightgrey')
        if 'x' in sci_axes:
            ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        if 'y' in sci_axes:
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        if legend:
            ax.legend(loc='upper left', prop=fontP)
        return fontP

    def plot_axes_map_detection_graph(self, data):
        time_data = data['acceleration_data_0']
        accel_data = data['acceleration_data_1']
        gravity = data['gravity']
        average_noise_intensity_label = data['average_noise_intensity_label']
        position_data = data['position_data']
        direction_vectors = data['direction_vectors']
        angle_errors = data['angle_errors']
        formatted_direction_vector = data['formatted_direction_vector']
        measurements = data['measurements']
        accel = data['accel']
        st_version = data['st_version']

        fig = plt.figure(figsize=(15, 7))
        gs = fig.add_gridspec(
            1, 2, width_ratios=[5, 3], bottom=0.080, top=0.840, left=0.055, right=0.960, hspace=0.166, wspace=0.060
        )
        ax_1 = fig.add_subplot(gs[0])
        ax_2 = fig.add_subplot(gs[1], projection='3d')

        # Add titles and logo
        try:
            filename = measurements[0]['name']
            dt = datetime.strptime(f'{filename.split("_")[2]} {filename.split("_")[3]}', '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X')
            if accel is not None:
                title_line2 += f' -- at {accel:0.0f} mm/s²'
        except Exception:
            title_line2 = measurements[0]['name'] + ' ...'
        title_lines = [
            {
                'x': 0.060,
                'y': 0.947,
                'text': 'AXES MAP CALIBRATION TOOL',
                'fontsize': 20,
                'color': self.KLIPPAIN_COLORS['purple'],
                'weight': 'bold',
            },
            {'x': 0.060, 'y': 0.939, 'va': 'top', 'text': title_line2},
            {'x': 0.50, 'y': 0.985, 'va': 'top', 'text': f'| Detected axes_map: {formatted_direction_vector}'},
        ]
        self.add_title(fig, title_lines)
        self.add_logo(fig)
        self.add_version_text(fig, st_version)

        # Plot acceleration data
        for i, (time, (accel_x, accel_y, accel_z)) in enumerate(zip(time_data, accel_data)):
            ax_1.plot(
                time,
                accel_x,
                label='X' if i == 0 else '',
                color=self.KLIPPAIN_COLORS['purple'],
                linewidth=0.5,
                zorder=50 if i == 0 else 10,
            )
            ax_1.plot(
                time,
                accel_y,
                label='Y' if i == 0 else '',
                color=self.KLIPPAIN_COLORS['orange'],
                linewidth=0.5,
                zorder=50 if i == 1 else 10,
            )
            ax_1.plot(
                time,
                accel_z,
                label='Z' if i == 0 else '',
                color=self.KLIPPAIN_COLORS['red_pink'],
                linewidth=0.5,
                zorder=50 if i == 2 else 10,
            )

        # Add gravity and noise level to a secondary legend
        ax_1_2 = ax_1.twinx()
        ax_1_2.yaxis.set_visible(False)
        ax_1_2.plot([], [], ' ', label=average_noise_intensity_label)
        ax_1_2.plot([], [], ' ', label=f'Measured gravity: {gravity / 1000:0.3f} m/s²')

        fontP = self.configure_axes(
            ax_1,
            xlabel='Time (s)',
            ylabel='Acceleration (mm/s²)',
            title='Acceleration (gravity offset removed)',
            sci_axes='y',
            legend=True,
        )
        ax_1_2.legend(loc='upper right', prop=fontP)

        # Plot 3D movement
        for i, ((position_x, position_y, position_z), average_direction_vector, angle_error) in enumerate(
            zip(position_data, direction_vectors, angle_errors)
        ):
            ax_2.plot(
                position_x, position_y, position_z, color=self.KLIPPAIN_COLORS['orange'], linestyle=':', linewidth=2
            )
            ax_2.scatter(position_x[0], position_y[0], position_z[0], color=self.KLIPPAIN_COLORS['red_pink'], zorder=10)
            ax_2.text(
                position_x[0] + 1,
                position_y[0],
                position_z[0],
                str(i + 1),
                color='black',
                fontsize=16,
                fontweight='bold',
                zorder=20,
            )

            # Plot average direction vector
            start_position = np.array([position_x[0], position_y[0], position_z[0]])
            end_position = start_position + average_direction_vector * np.linalg.norm(
                [position_x[-1] - position_x[0], position_y[-1] - position_y[0], position_z[-1] - position_z[0]]
            )
            ax_2.plot(
                [start_position[0], end_position[0]],
                [start_position[1], end_position[1]],
                [start_position[2], end_position[2]],
                label=f'{["X", "Y", "Z"][i]} angle: {angle_error:0.2f}°',
                color=self.KLIPPAIN_COLORS['purple'],
                linestyle='-',
                linewidth=2,
            )

            self.configure_axes(
                ax_2,
                xlabel='X Position (mm)',
                ylabel='Y Position (mm)',
                zlabel='Z Position (mm)',
                title='Estimated movement in 3D space',
                legend=True,
            )

        return fig

    def plot_static_frequency_graph(self, data):
        freq = data['freq']
        duration = data['duration']
        accel_per_hz = data['accel_per_hz']
        st_version = data['st_version']
        measurements = data['measurements']
        t = data['t']
        bins = data['bins']
        pdata = data['pdata']
        max_freq = data['max_freq']

        fig, axes = plt.subplots(
            1,
            2,
            gridspec_kw={
                'width_ratios': [5, 3],
                'bottom': 0.080,
                'top': 0.840,
                'left': 0.050,
                'right': 0.985,
                'hspace': 0.166,
                'wspace': 0.138,
            },
            figsize=(15, 7),
        )
        ax_1, ax_2 = axes

        # Add titles and logo
        try:
            filename_parts = measurements[0]['name'].split('_')
            dt = datetime.strptime(f'{filename_parts[2]} {filename_parts[3]}', '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X') + ' -- ' + filename_parts[1].upper() + ' axis'
        except Exception:
            title_line2 = measurements[0]['name']
        title_line3 = f'| Maintained frequency: {freq}Hz' if freq is not None else ''
        title_line3 += f' for {duration}s' if duration is not None and title_line3 != '' else ''
        title_lines = [
            {
                'x': 0.060,
                'y': 0.947,
                'text': 'STATIC FREQUENCY HELPER TOOL',
                'fontsize': 20,
                'color': self.KLIPPAIN_COLORS['purple'],
                'weight': 'bold',
            },
            {'x': 0.060, 'y': 0.939, 'va': 'top', 'text': title_line2},
            {'x': 0.55, 'y': 0.985, 'va': 'top', 'fontsize': 14, 'text': title_line3},
            {
                'x': 0.55,
                'y': 0.950,
                'va': 'top',
                'fontsize': 11,
                'text': f'| Accel per Hz used: {accel_per_hz} mm/s²/Hz' if accel_per_hz is not None else '',
            },
        ]
        self.add_title(fig, title_lines)
        self.add_logo(fig)
        self.add_version_text(fig, st_version)

        # Plot spectrogram
        vmin_value = np.percentile(pdata, self.SPECTROGRAM_LOW_PERCENTILE_FILTER)
        ax_1.imshow(
            pdata.T,
            norm=matplotlib.colors.LogNorm(vmin=vmin_value),
            cmap='inferno',
            aspect='auto',
            extent=[t[0], t[-1], bins[0], bins[-1]],
            origin='lower',
            interpolation='antialiased',
        )
        ax_1.set_xlim([0.0, max_freq])
        self.configure_axes(
            ax_1, xlabel='Frequency (Hz)', ylabel='Time (s)', grid=False, title='Time-Frequency Spectrogram'
        )

        # Plot cumulative energy
        ax_2.plot(np.trapz(pdata, t, axis=0), bins, color=self.KLIPPAIN_COLORS['orange'])
        ax_2.set_ylim([bins[0], bins[-1]])
        self.configure_axes(
            ax_2, xlabel='Cumulative Energy', ylabel='Time (s)', sci_axes='x', title='Vibrations', legend=False
        )

        return fig

    def plot_belts_graph(self, data):
        signal1 = data['signal1']
        signal2 = data['signal2']
        similarity_factor = data['similarity_factor']
        mhi = data['mhi']
        signal1_belt = data['signal1_belt']
        signal2_belt = data['signal2_belt']
        kinematics = data['kinematics']
        mode, _, _, accel_per_hz, _, sweeping_accel, sweeping_period = data['test_params']
        st_version = data['st_version']
        measurements = data['measurements']
        max_freq = data['max_freq']
        max_scale = data['max_scale']

        fig, axes = plt.subplots(
            1,
            2,
            gridspec_kw={
                'width_ratios': [5, 3],
                'bottom': 0.080,
                'top': 0.840,
                'left': 0.050,
                'right': 0.985,
                'hspace': 0.166,
                'wspace': 0.138,
            },
            figsize=(15, 7),
        )
        ax_1, ax_2 = axes

        # Add titles and logo
        try:
            filename = measurements[0]['name']
            dt = datetime.strptime(f'{filename.split("_")[2]} {filename.split("_")[3]}', '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X')
            if kinematics is not None:
                title_line2 += ' -- ' + kinematics.upper() + ' kinematics'
        except Exception:
            title_line2 = measurements[0]['name'] + ' / ' + measurements[1]['name']

        title_line3 = f'| Mode: {mode}'
        title_line3 += f' -- ApH: {accel_per_hz}' if accel_per_hz is not None else ''
        if mode == 'SWEEPING':
            title_line3 += f' [sweeping period: {sweeping_period} s - accel: {sweeping_accel} mm/s²]'

        title_lines = [
            {
                'x': 0.060,
                'y': 0.947,
                'text': 'RELATIVE BELTS CALIBRATION TOOL',
                'fontsize': 20,
                'color': self.KLIPPAIN_COLORS['purple'],
                'weight': 'bold',
            },
            {'x': 0.060, 'y': 0.939, 'va': 'top', 'text': title_line2},
            {
                'x': 0.481,
                'y': 0.985,
                'va': 'top',
                'fontsize': 10,
                'text': title_line3,
            },
        ]

        if kinematics in {'limited_corexy', 'corexy', 'limited_corexz', 'corexz'}:
            title_lines.extend(
                [
                    {
                        'x': 0.480,
                        'y': 0.953,
                        'va': 'top',
                        'fontsize': 13,
                        'text': f'| Estimated similarity: {similarity_factor:.1f}%',
                    },
                    {'x': 0.480, 'y': 0.920, 'va': 'top', 'fontsize': 13, 'text': f'| {mhi} (experimental)'},
                ]
            )

        self.add_title(fig, title_lines)
        self.add_logo(fig)
        self.add_version_text(fig, st_version)

        # Plot PSD signals
        ax_1.plot(signal1.freqs, signal1.psd, label='Belt ' + signal1_belt, color=self.KLIPPAIN_COLORS['orange'])
        ax_1.plot(signal2.freqs, signal2.psd, label='Belt ' + signal2_belt, color=self.KLIPPAIN_COLORS['purple'])
        psd_highest_max = max(signal1.psd.max(), signal2.psd.max())
        ax_1.set_xlim([0, max_freq])
        ax_1.set_ylim([0, max_scale if max_scale is not None else psd_highest_max * 1.1])

        # Annotate peaks
        paired_peak_count = 0
        unpaired_peak_count = 0
        offsets_table_data = []
        for _, (peak1, peak2) in enumerate(signal1.paired_peaks):
            label = self.ALPHABET[paired_peak_count]
            amplitude_offset = abs(((signal2.psd[peak2[0]] - signal1.psd[peak1[0]]) / psd_highest_max) * 100)
            frequency_offset = abs(signal2.freqs[peak2[0]] - signal1.freqs[peak1[0]])
            offsets_table_data.append([f'Peaks {label}', f'{frequency_offset:.1f} Hz', f'{amplitude_offset:.1f} %'])

            ax_1.plot(signal1.freqs[peak1[0]], signal1.psd[peak1[0]], 'x', color='black')
            ax_1.plot(signal2.freqs[peak2[0]], signal2.psd[peak2[0]], 'x', color='black')
            ax_1.plot(
                [signal1.freqs[peak1[0]], signal2.freqs[peak2[0]]],
                [signal1.psd[peak1[0]], signal2.psd[peak2[0]]],
                ':',
                color='gray',
            )

            ax_1.annotate(
                label + '1',
                (signal1.freqs[peak1[0]], signal1.psd[peak1[0]]),
                textcoords='offset points',
                xytext=(8, 5),
                ha='left',
                fontsize=13,
                color='black',
            )
            ax_1.annotate(
                label + '2',
                (signal2.freqs[peak2[0]], signal2.psd[peak2[0]]),
                textcoords='offset points',
                xytext=(8, 5),
                ha='left',
                fontsize=13,
                color='black',
            )
            paired_peak_count += 1

        for peak in signal1.unpaired_peaks:
            ax_1.plot(signal1.freqs[peak], signal1.psd[peak], 'x', color='black')
            ax_1.annotate(
                str(unpaired_peak_count + 1),
                (signal1.freqs[peak], signal1.psd[peak]),
                textcoords='offset points',
                xytext=(8, 5),
                ha='left',
                fontsize=13,
                color='red',
                weight='bold',
            )
            unpaired_peak_count += 1

        for peak in signal2.unpaired_peaks:
            ax_1.plot(signal2.freqs[peak], signal2.psd[peak], 'x', color='black')
            ax_1.annotate(
                str(unpaired_peak_count + 1),
                (signal2.freqs[peak], signal2.psd[peak]),
                textcoords='offset points',
                xytext=(8, 5),
                ha='left',
                fontsize=13,
                color='red',
                weight='bold',
            )
            unpaired_peak_count += 1

        # Add estimated similarity to the graph on a secondary legend
        ax_1_2 = ax_1.twinx()
        ax_1_2.yaxis.set_visible(False)
        ax_1_2.plot([], [], ' ', label=f'Number of unpaired peaks: {unpaired_peak_count}')

        fontP = self.configure_axes(
            ax_1,
            xlabel='Frequency (Hz)',
            ylabel='Power spectral density',
            title='Belts frequency profiles',
            sci_axes='y',
            legend=True,
        )
        ax_1_2.legend(loc='upper right', prop=fontP)

        # Print the table of offsets
        if len(offsets_table_data) > 0:
            columns = ['', 'Frequency delta', 'Amplitude delta']
            offset_table = ax_1.table(
                cellText=offsets_table_data,
                colLabels=columns,
                bbox=[0.66, 0.79, 0.33, 0.15],
                loc='upper right',
                cellLoc='center',
            )
            offset_table.auto_set_font_size(False)
            offset_table.set_fontsize(8)
            offset_table.auto_set_column_width([0, 1, 2])
            offset_table.set_zorder(100)
            for cell in offset_table.get_celld().values():
                cell.set_facecolor('white')
                cell.set_alpha(0.6)

        # Plot cross-belts comparison
        max_psd = max(np.max(signal1.psd), np.max(signal2.psd))
        ideal_line = np.linspace(0, max_psd * 1.1, 500)
        green_boundary = ideal_line + (0.35 * max_psd * np.exp(-ideal_line / (0.6 * max_psd)))
        ax_2.fill_betweenx(ideal_line, ideal_line, green_boundary, color='green', alpha=0.15)
        ax_2.fill_between(ideal_line, ideal_line, green_boundary, color='green', alpha=0.15, label='Good zone')
        ax_2.plot(
            ideal_line,
            ideal_line,
            '--',
            label='Ideal line',
            color='red',
            linewidth=2,
        )

        ax_2.plot(signal1.psd, signal2.psd, color='dimgrey', marker='o', markersize=1.5)
        ax_2.fill_betweenx(signal2.psd, signal1.psd, color=self.KLIPPAIN_COLORS['red_pink'], alpha=0.1)

        # Annotate peaks
        paired_peak_count = 0
        unpaired_peak_count = 0
        for _, (peak1, peak2) in enumerate(signal1.paired_peaks):
            label = self.ALPHABET[paired_peak_count]
            freq1 = signal1.freqs[peak1[0]]
            freq2 = signal2.freqs[peak2[0]]

            if abs(freq1 - freq2) < 1:
                ax_2.plot(signal1.psd[peak1[0]], signal2.psd[peak2[0]], marker='o', color='black', markersize=7)
                ax_2.annotate(
                    f'{label}1/{label}2',
                    (signal1.psd[peak1[0]], signal2.psd[peak2[0]]),
                    textcoords='offset points',
                    xytext=(-7, 7),
                    fontsize=13,
                    color='black',
                )
            else:
                ax_2.plot(
                    signal1.psd[peak2[0]],
                    signal2.psd[peak2[0]],
                    marker='o',
                    color=self.KLIPPAIN_COLORS['purple'],
                    markersize=7,
                )
                ax_2.plot(
                    signal1.psd[peak1[0]],
                    signal2.psd[peak1[0]],
                    marker='o',
                    color=self.KLIPPAIN_COLORS['orange'],
                    markersize=7,
                )
                ax_2.annotate(
                    f'{label}1',
                    (signal1.psd[peak1[0]], signal2.psd[peak1[0]]),
                    textcoords='offset points',
                    xytext=(0, 7),
                    fontsize=13,
                    color='black',
                )
                ax_2.annotate(
                    f'{label}2',
                    (signal1.psd[peak2[0]], signal2.psd[peak2[0]]),
                    textcoords='offset points',
                    xytext=(0, 7),
                    fontsize=13,
                    color='black',
                )
            paired_peak_count += 1

        for _, peak_index in enumerate(signal1.unpaired_peaks):
            ax_2.plot(
                signal1.psd[peak_index],
                signal2.psd[peak_index],
                marker='o',
                color=self.KLIPPAIN_COLORS['orange'],
                markersize=7,
            )
            ax_2.annotate(
                str(unpaired_peak_count + 1),
                (signal1.psd[peak_index], signal2.psd[peak_index]),
                textcoords='offset points',
                fontsize=13,
                weight='bold',
                color=self.KLIPPAIN_COLORS['red_pink'],
                xytext=(0, 7),
            )
            unpaired_peak_count += 1

        for _, peak_index in enumerate(signal2.unpaired_peaks):
            ax_2.plot(
                signal1.psd[peak_index],
                signal2.psd[peak_index],
                marker='o',
                color=self.KLIPPAIN_COLORS['purple'],
                markersize=7,
            )
            ax_2.annotate(
                str(unpaired_peak_count + 1),
                (signal1.psd[peak_index], signal2.psd[peak_index]),
                textcoords='offset points',
                fontsize=13,
                weight='bold',
                color=self.KLIPPAIN_COLORS['red_pink'],
                xytext=(0, 7),
            )
            unpaired_peak_count += 1

        ax_2.set_xlim([0, max_psd * 1.1])
        ax_2.set_ylim([0, max_psd * 1.1])
        self.configure_axes(
            ax_2,
            xlabel=f'Belt {signal1_belt}',
            ylabel=f'Belt {signal2_belt}',
            title='Cross-belts comparison plot',
            sci_axes='xy',
            legend=True,
        )

        return fig

    def plot_input_shaper_graph(self, data):
        measurements = data['measurements']
        compat = data['compat']
        max_smoothing_computed = data['max_smoothing_computed']
        max_freq = data['max_freq']
        calibration_data = data['calibration_data']
        shapers = data['shapers']
        shaper_table_data = data['shaper_table_data']
        shaper_choices = data['shaper_choices']
        peaks = data['peaks']
        peaks_freqs = data['peaks_freqs']
        peaks_threshold = data['peaks_threshold']
        fr = data['fr']
        zeta = data['zeta']
        t = data['t']
        bins = data['bins']
        pdata = data['pdata']
        # shapers_tradeoff_data = data['shapers_tradeoff_data']
        mode, _, _, accel_per_hz, _, sweeping_accel, sweeping_period = data['test_params']
        max_smoothing = data['max_smoothing']
        scv = data['scv']
        st_version = data['st_version']
        max_scale = data['max_scale']

        fig = plt.figure(figsize=(15, 11.6))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[4, 3],
            width_ratios=[5, 4],
            bottom=0.050,
            top=0.890,
            left=0.048,
            right=0.966,
            hspace=0.169,
            wspace=0.150,
        )
        ax_1 = fig.add_subplot(gs[0, 0])
        ax_2 = fig.add_subplot(gs[1, 0])
        ax_3 = fig.add_subplot(gs[1, 1])

        # Add titles and logo
        try:
            filename_parts = measurements[0]['name'].split('_')
            dt = datetime.strptime(f'{filename_parts[2]} {filename_parts[3]}', '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X') + ' -- ' + filename_parts[1].upper() + ' axis'
            if compat:
                title_line3 = '| Older Klipper version detected, damping ratio'
                title_line4 = '| and SCV are not used for filter recommendations!'
            else:
                max_smoothing_string = (
                    f'default (={max_smoothing_computed:0.3f})' if max_smoothing is None else f'{max_smoothing:0.3f}'
                )
                title_line3 = f'| Square corner velocity: {scv} mm/s'
                title_line4 = f'| Allowed smoothing: {max_smoothing_string}'
        except Exception:
            title_line2 = measurements[0]['name']
            title_line3 = ''
            title_line4 = ''
        title_line5 = f'| Mode: {mode}'
        title_line5 += f' -- ApH: {accel_per_hz}' if accel_per_hz is not None else ''
        if mode == 'SWEEPING':
            title_line5 += f' [sweeping period: {sweeping_period} s - accel: {sweeping_accel} mm/s²]'
        title_lines = [
            {
                'x': 0.065,
                'y': 0.965,
                'text': 'INPUT SHAPER CALIBRATION TOOL',
                'fontsize': 20,
                'color': self.KLIPPAIN_COLORS['purple'],
                'weight': 'bold',
            },
            {'x': 0.065, 'y': 0.957, 'va': 'top', 'text': title_line2},
            {'x': 0.481, 'y': 0.990, 'va': 'top', 'fontsize': 11, 'text': title_line5},
            {'x': 0.480, 'y': 0.970, 'va': 'top', 'fontsize': 14, 'text': title_line3},
            {'x': 0.480, 'y': 0.949, 'va': 'top', 'fontsize': 14, 'text': title_line4},
        ]
        self.add_title(fig, title_lines)
        self.add_logo(fig, position=[0.001, 0.924, 0.075, 0.075])
        self.add_version_text(fig, st_version, position=(0.995, 0.985))

        # Plot Frequency Profile
        freqs = calibration_data.freqs
        psd = calibration_data.psd_sum
        px = calibration_data.psd_x
        py = calibration_data.psd_y
        pz = calibration_data.psd_z

        ax_1.plot(freqs, psd, label='X+Y+Z', color='purple', zorder=5)
        ax_1.plot(freqs, px, label='X', color='red')
        ax_1.plot(freqs, py, label='Y', color='green')
        ax_1.plot(freqs, pz, label='Z', color='blue')
        ax_1.set_xlim([0, max_freq])
        ax_1.set_ylim([0, max_scale if max_scale is not None else psd.max() * 1.05])

        ax_1_2 = ax_1.twinx()
        ax_1_2.yaxis.set_visible(False)
        for shaper in shapers:
            ax_1_2.plot(freqs, shaper.vals, label=shaper.name.upper(), linestyle='dotted')

        # Draw shaper filtered PSDs and add their specific parameters in the legend
        for shaper in shaper_table_data['shapers']:
            if shaper['type'] == shaper_choices[0]:
                ax_1.plot(freqs, psd * shaper['vals'], label=f'With {shaper_choices[0]} applied', color='cyan')
            if len(shaper_choices) > 1 and shaper['type'] == shaper_choices[1]:
                ax_1.plot(freqs, psd * shaper['vals'], label=f'With {shaper_choices[1]} applied', color='lime')

        # Draw detected peaks and their respective labels
        ax_1.plot(peaks_freqs, psd[peaks], 'x', color='black', markersize=8)
        for idx, peak in enumerate(peaks):
            fontcolor = 'red' if psd[peak] > peaks_threshold[1] else 'black'
            fontweight = 'bold' if psd[peak] > peaks_threshold[1] else 'normal'
            ax_1.annotate(
                f'{idx + 1}',
                (freqs[peak], psd[peak]),
                textcoords='offset points',
                xytext=(8, 5),
                ha='left',
                fontsize=13,
                color=fontcolor,
                weight=fontweight,
            )
        ax_1.axhline(y=peaks_threshold[0], color='black', linestyle='--', linewidth=0.5)
        ax_1.axhline(y=peaks_threshold[1], color='black', linestyle='--', linewidth=0.5)
        ax_1.fill_between(freqs, 0, peaks_threshold[0], color='green', alpha=0.15, label='Relax Region')
        ax_1.fill_between(
            freqs, peaks_threshold[0], peaks_threshold[1], color='orange', alpha=0.2, label='Warning Region'
        )

        fontP = self.configure_axes(
            ax_1,
            xlabel='Frequency (Hz)',
            ylabel='Power spectral density',
            title=f'Axis Frequency Profile (ω0={fr:.1f}Hz, ζ={zeta:.3f})',
            sci_axes='y',
            legend=True,
        )
        ax_1_2.legend(loc='upper right', prop=fontP)

        # Plot a time-frequency spectrogram.
        # This can highlight hidden spots from the standard PSD graph(like a running fan or aliasing accelerometer)
        # Note: We need to normalize the data to get a proper signal on the spectrogram... But using "LogNorm" provide
        # too much background noise and using "Normalize" make only the resonnance appearing and hide interesting elements
        # So we need to filter out the lower part of the data (ie. find the proper vmin for LogNorm) by taking a percentile
        # Noreover, we use imgshow() that is better than pcolormesh (for a png image generation) since it's already
        # rasterized. This allow to save ~150-200MB of RAM during the "fig.savefig()" operation a is a bit faster.
        vmin_value = np.percentile(pdata, self.SPECTROGRAM_LOW_PERCENTILE_FILTER)
        ax_2.imshow(
            pdata.T,
            norm=matplotlib.colors.LogNorm(vmin=vmin_value),
            cmap='inferno',
            aspect='auto',
            extent=[t[0], t[-1], bins[0], bins[-1]],
            origin='lower',
            interpolation='antialiased',
        )

        # Add peaks lines in the spectrogram to get hint from peaks found in the first graph
        for idx, peak in enumerate(peaks_freqs):
            ax_2.axvline(peak, color='cyan', linestyle='dotted', linewidth=1)
            ax_2.annotate(
                f'Peak {idx + 1} ({peak:.1f} Hz)',
                (peak, bins[-1] * 0.9),
                textcoords='data',
                color='cyan',
                rotation=90,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
            )

        ax_2.set_xlim([0.0, max_freq])
        self.configure_axes(
            ax_2, xlabel='Frequency (Hz)', ylabel='Time (s)', title='Time-Frequency Spectrogram', grid=False
        )

        # TODO: re-add this in next release
        # --------------------------------------------------------------------------------------------------------------
        ax_3.remove()
        # --------------------------------------------------------------------------------------------------------------
        # # Plot the vibrations vs acceleration curves for each shaper
        # max_shaper_accel = 0
        # for shaper_name, data in shapers_tradeoff_data.items():
        #     ax_3.plot(data['accels'], data['vibrs'], label=shaper_name, zorder=10)
        #     # find the accel of the same shaper in the standard k_shapers and use it as max_shaper_accel
        #     shaper = next(s for s in shapers if s.name == shaper_name)
        #     max_shaper_accel = max(max_shaper_accel, shaper.max_accel)

        # # Configure the main axes
        # ax_3.set_xlim([0, max_shaper_accel * 1.75])  # ~2x of the standard shapers (to see higher accels)
        # ax_3.set_ylim([0, None])
        # ax_3.legend(loc='best', prop=fontP)
        # self.configure_axes(
        #     ax_3,
        #     xlabel='Acceleration (mm/s²)',
        #     ylabel='Remaining Vibrations (%)',
        #     title='Filters Performance',
        #     legend=False,  # Legend is configured to be at the "best" location
        # )

        # --------------------------------------------------------------------------------------------------------------
        # Plot the vibrations vs acceleration curves for each shaper
        # ax_3.set_title('Remaining Vibrations vs Acceleration and Smoothing')
        # shaper_name = 'mzv'  # Choose the shaper you're interested in
        # shaper_data = shapers_tradeoff_data[shaper_name]

        # # Prepare data for the heatmap
        # X, Y = np.meshgrid(shaper_data['smoothings'], shaper_data['accels'])
        # Z = shaper_data['vibrations_grid']

        # # Create the heatmap
        # heatmap = ax_3.pcolormesh(Y, X, Z, shading='gouraud', cmap='inferno', vmin=0, vmax=100)
        # fig.colorbar(heatmap, ax=ax_3, label='Remaining Vibrations (%)')

        # ax_3.set_xlabel('Acceleration (mm/s²)')
        # ax_3.set_ylabel('Smoothing (mm/s²)')
        # # ax_3.set_xlim([shaper_data['smoothings'][0], shaper_data['smoothings'][-1]])
        # ax_3.set_ylim([0.0, None])
        # # ax_3.set_ylim([shaper_data['accels'][0], shaper_data['accels'][-1]])
        # ax_3.set_xlim([0.0, None])

        # # Optionally, overlay contours for specific vibration levels
        # contours = ax_3.contour(Y, X, Z, levels=[5, 10, 20, 30], colors='white', linewidths=0.5)
        # ax_3.clabel(contours, inline=True, fontsize=8, fmt='%1.0f%%')

        # ax_3.set_title(f'Heatmap of Remaining Vibrations for {shaper_name.upper()}')
        # --------------------------------------------------------------------------------------------------------------

        # Print shaper table
        columns = ['Type', 'Frequency', 'Vibrations', 'Smoothing', 'Max Accel']
        table_data = [
            [
                shaper['type'].upper(),
                f'{shaper["frequency"]:.1f} Hz',
                f'{shaper["vibrations"] * 100:.1f} %',
                f'{shaper["smoothing"]:.3f}',
                f'{round(shaper["max_accel"] / 10) * 10:.0f}',
            ]
            for shaper in shaper_table_data['shapers']
        ]

        table = plt.table(cellText=table_data, colLabels=columns, bbox=[1.100, 0.535, 0.830, 0.240], cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2, 3, 4])
        table.set_zorder(100)
        bold_font = matplotlib.font_manager.FontProperties(weight='bold')
        for key, cell in table.get_celld().items():
            row, col = key
            cell.set_text_props(ha='center', va='center')
            if col == 0:
                cell.get_text().set_fontproperties(bold_font)
                cell.get_text().set_color(self.KLIPPAIN_COLORS['dark_purple'])
            if row == 0:
                cell.get_text().set_fontproperties(bold_font)
                cell.get_text().set_color(self.KLIPPAIN_COLORS['dark_orange'])

        # Add the filter general recommendations and estimated damping ratio
        fig.text(
            0.575,
            0.897,
            'Recommended filters:',
            fontsize=15,
            fontweight='bold',
            color=self.KLIPPAIN_COLORS['dark_purple'],
        )
        recommendations = shaper_table_data['recommendations']
        for idx, rec in enumerate(recommendations):
            fig.text(0.580, 0.867 - idx * 0.025, rec, fontsize=14, color=self.KLIPPAIN_COLORS['purple'])
        new_idx = len(recommendations)
        fig.text(
            0.580,
            0.867 - new_idx * 0.025,
            f'    -> Estimated damping ratio (ζ): {shaper_table_data["damping_ratio"]:.3f}',
            fontsize=14,
            color=self.KLIPPAIN_COLORS['purple'],
        )

        return fig

    def plot_vibrations_graph(self, data):
        measurements = data['measurements']
        all_speeds = data['all_speeds']
        all_angles = data['all_angles']
        all_angles_energy = data['all_angles_energy']
        good_speeds = data['good_speeds']
        good_angles = data['good_angles']
        kinematics = data['kinematics']
        accel = data['accel']
        motors = data['motors']
        motors_config_differences = data['motors_config_differences']
        symmetry_factor = data['symmetry_factor']
        spectrogram_data = data['spectrogram_data']
        sp_min_energy = data['sp_min_energy']
        sp_max_energy = data['sp_max_energy']
        sp_variance_energy = data['sp_variance_energy']
        vibration_metric = data['vibration_metric']
        num_peaks = data['num_peaks']
        vibration_peaks = data['vibration_peaks']
        target_freqs = data['target_freqs']
        main_angles = data['main_angles']
        global_motor_profile = data['global_motor_profile']
        motor_profiles = data['motor_profiles']
        max_freq = data['max_freq']
        motor_fr = data['motor_fr']
        motor_zeta = data['motor_zeta']
        motor_res_idx = data['motor_res_idx']
        st_version = data['st_version']

        fig = plt.figure(figsize=(20, 11.5))
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[1, 1],
            width_ratios=[4, 8, 6],
            bottom=0.050,
            top=0.890,
            left=0.040,
            right=0.985,
            hspace=0.166,
            wspace=0.138,
        )
        ax_1 = fig.add_subplot(gs[0, 0], projection='polar')
        ax_4 = fig.add_subplot(gs[1, 0], projection='polar')
        ax_2 = fig.add_subplot(gs[0, 1])
        ax_3 = fig.add_subplot(gs[0, 2])
        ax_5 = fig.add_subplot(gs[1, 1])
        ax_6 = fig.add_subplot(gs[1, 2])

        # Add title
        try:
            filename_parts = measurements[0]['name'].split('_')
            dt = datetime.strptime(f'{filename_parts[4]} {filename_parts[5].split("-")[0]}', '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X')
            if accel is not None:
                title_line2 += f' at {accel} mm/s² -- {kinematics.upper()} kinematics'
        except Exception:
            title_line2 = measurements[0]['name']
        title_lines = [
            {
                'x': 0.060,
                'y': 0.965,
                'text': 'MACHINE VIBRATIONS ANALYSIS TOOL',
                'fontsize': 20,
                'color': self.KLIPPAIN_COLORS['purple'],
                'weight': 'bold',
            },
            {'x': 0.060, 'y': 0.957, 'va': 'top', 'text': title_line2},
        ]
        self.add_title(fig, title_lines)
        self.add_logo(fig, position=[0.001, 0.924, 0.075, 0.075])
        self.add_version_text(fig, st_version, position=(0.995, 0.985))

        # Plot the motors infos to the top of the graph if they are detected / specified (not mandatory for CLI mode)
        if motors is not None and len(motors) == 2:
            motor_details = [(motors[0], 'X motor'), (motors[1], 'Y motor')]
            distance = 0.27 if motors[0].get_config('autotune_enabled') else 0.16
            if motors[0].get_config('autotune_enabled'):
                config_blocks = [
                    f'| {lbl}: {mot.get_config("motor").upper()} on {mot.get_config("tmc").upper()} @ {mot.get_config("voltage"):0.1f}V {mot.get_config("run_current"):0.2f}A - {mot.get_config("microsteps")}usteps'
                    for mot, lbl in motor_details
                ]
                config_blocks.append(
                    f'| TMC Autotune enabled (PWM freq target: X={int(motors[0].get_config("pwm_freq_target") / 1000)}kHz / Y={int(motors[1].get_config("pwm_freq_target") / 1000)}kHz)'
                )
            else:
                config_blocks = [
                    f'| {lbl}: {mot.get_config("tmc").upper()} @ {mot.get_config("run_current"):0.2f}A - {mot.get_config("microsteps")}usteps'
                    for mot, lbl in motor_details
                ]
                config_blocks.append('| TMC Autotune not detected')
            for idx, block in enumerate(config_blocks):
                fig.text(
                    0.41,
                    0.990 - 0.015 * idx,
                    block,
                    ha='left',
                    va='top',
                    fontsize=10,
                    color=self.KLIPPAIN_COLORS['dark_purple'],
                )
            tmc_registers = motors[0].get_registers()
            idx = -1
            for idx, (register, settings) in enumerate(tmc_registers.items()):
                settings_str = ' '.join(f'{k}={v}' for k, v in settings.items())
                tmc_block = f'| {register.upper()}: {settings_str}'
                fig.text(
                    0.41 + distance,
                    0.990 - 0.015 * idx,
                    tmc_block,
                    ha='left',
                    va='top',
                    fontsize=10,
                    color=self.KLIPPAIN_COLORS['dark_purple'],
                )
            if motors_config_differences is not None:
                differences_text = f'| Y motor diff: {motors_config_differences}'
                fig.text(
                    0.41 + distance,
                    0.990 - 0.015 * (idx + 1),
                    differences_text,
                    ha='left',
                    va='top',
                    fontsize=10,
                    color=self.KLIPPAIN_COLORS['dark_purple'],
                )

        # Plot angle energy profile (Polar plot)
        angles_radians = np.deg2rad(all_angles)
        ymax = all_angles_energy.max() * 1.05
        ax_1.plot(angles_radians, all_angles_energy, color=self.KLIPPAIN_COLORS['purple'], zorder=5)
        ax_1.fill(angles_radians, all_angles_energy, color=self.KLIPPAIN_COLORS['purple'], alpha=0.3)
        ax_1.set_xlim([0, np.deg2rad(360)])
        ax_1.set_ylim([0, ymax])
        ax_1.set_theta_zero_location('E')
        ax_1.set_theta_direction(1)
        ax_1.set_thetagrids([theta * 15 for theta in range(360 // 15)])
        ax_1.text(
            0,
            0,
            f'Symmetry: {symmetry_factor:.1f}%',
            ha='center',
            va='center',
            color=self.KLIPPAIN_COLORS['red_pink'],
            fontsize=12,
            fontweight='bold',
            zorder=6,
        )

        for _, (start, end, _) in enumerate(good_angles):
            ax_1.axvline(
                angles_radians[start],
                all_angles_energy[start] / ymax,
                color=self.KLIPPAIN_COLORS['red_pink'],
                linestyle='dotted',
                linewidth=1.5,
            )
            ax_1.axvline(
                angles_radians[end],
                all_angles_energy[end] / ymax,
                color=self.KLIPPAIN_COLORS['red_pink'],
                linestyle='dotted',
                linewidth=1.5,
            )
            ax_1.fill_between(
                angles_radians[start:end],
                all_angles_energy[start:end],
                all_angles_energy.max() * 1.05,
                color='green',
                alpha=0.2,
            )

        self.configure_axes(ax_1, title='Polar angle energy profile')

        # Polar plot doesn't follow the gridspec margin, so we adjust it manually here
        pos = ax_1.get_position()
        new_pos = [pos.x0 - 0.01, pos.y0 - 0.01, pos.width, pos.height]
        ax_1.set_position(new_pos)

        # Plot polar vibrations heatmap
        # Note: Assuming speeds defines the radial distance from the center, we need to create a meshgrid
        # for both angles and speeds to map the spectrogram data onto a polar plot correctly
        radius, theta = np.meshgrid(all_speeds, angles_radians)
        ax_4.pcolormesh(
            theta, radius, spectrogram_data, norm=matplotlib.colors.LogNorm(), cmap='inferno', shading='auto'
        )
        ax_4.set_theta_zero_location('E')
        ax_4.set_theta_direction(1)
        ax_4.set_thetagrids([theta * 15 for theta in range(360 // 15)])
        ax_4.set_ylim([0, max(all_speeds)])
        self.configure_axes(ax_4, title='Polar vibrations heatmap', grid=False)

        ax_4.tick_params(axis='y', which='both', colors='white', labelsize='medium')

        # Polar plot doesn't follow the gridspec margin, so we adjust it manually here
        pos = ax_4.get_position()
        new_pos = [pos.x0 - 0.01, pos.y0 - 0.01, pos.width, pos.height]
        ax_4.set_position(new_pos)

        # Plot global speed energy profile
        ax_2.plot(all_speeds, sp_min_energy, label='Minimum', color=self.KLIPPAIN_COLORS['dark_purple'], zorder=5)
        ax_2.plot(all_speeds, sp_max_energy, label='Maximum', color=self.KLIPPAIN_COLORS['purple'], zorder=5)
        ax_2.plot(
            all_speeds,
            sp_variance_energy,
            label='Variance',
            color=self.KLIPPAIN_COLORS['orange'],
            zorder=5,
            linestyle='--',
        )
        ax_2.set_xlim([all_speeds.min(), all_speeds.max()])
        ax_2.set_ylim([0, sp_max_energy.max() * 1.15])

        # Add a secondary axis to plot the vibration metric
        ax_2_2 = ax_2.twinx()
        ax_2_2.yaxis.set_visible(False)
        ax_2_2.plot(
            all_speeds,
            vibration_metric,
            label=f'Vibration metric ({num_peaks} bad peaks)',
            color=self.KLIPPAIN_COLORS['red_pink'],
            zorder=5,
        )
        ax_2_2.set_ylim([-(vibration_metric.max() * 0.025), vibration_metric.max() * 1.07])

        if vibration_peaks is not None and len(vibration_peaks) > 0:
            ax_2_2.plot(
                all_speeds[vibration_peaks],
                vibration_metric[vibration_peaks],
                'x',
                color='black',
                markersize=8,
                zorder=10,
            )
            for idx, peak in enumerate(vibration_peaks):
                ax_2_2.annotate(
                    f'{idx + 1}',
                    (all_speeds[peak], vibration_metric[peak]),
                    textcoords='offset points',
                    xytext=(5, 5),
                    fontweight='bold',
                    ha='left',
                    fontsize=13,
                    color=self.KLIPPAIN_COLORS['red_pink'],
                    zorder=10,
                )
        for idx, (start, end, _) in enumerate(good_speeds):
            ax_2_2.fill_between(
                all_speeds[start:end],
                -(vibration_metric.max() * 0.025),
                vibration_metric[start:end],
                color='green',
                alpha=0.2,
                label=f'Zone {idx + 1}: {all_speeds[start]:.1f} to {all_speeds[end]:.1f} mm/s',
            )

        fontP = self.configure_axes(
            ax_2, xlabel='Speed (mm/s)', ylabel='Energy', title='Global speed energy profile', legend=True
        )
        ax_2_2.legend(loc='upper right', prop=fontP)

        # Plot the angular speed energy profiles
        angle_settings = {
            0: ('X (0 deg)', 'purple', 10),
            90: ('Y (90 deg)', 'dark_purple', 5),
            45: ('A (45 deg)' if kinematics in {'corexy', 'limited_corexy'} else '45 deg', 'orange', 10),
            135: ('B (135 deg)' if kinematics in {'corexy', 'limited_corexy'} else '135 deg', 'dark_orange', 5),
        }
        for angle, (label, color, zorder) in angle_settings.items():
            idx = np.searchsorted(all_angles, angle, side='left')
            ax_3.plot(all_speeds, spectrogram_data[idx], label=label, color=self.KLIPPAIN_COLORS[color], zorder=zorder)

        ax_3.set_xlim([all_speeds.min(), all_speeds.max()])
        max_value = max(spectrogram_data[angle].max() for angle in angle_settings.keys())
        ax_3.set_ylim([0, max_value * 1.1])
        fontP = self.configure_axes(
            ax_3, xlabel='Speed (mm/s)', ylabel='Energy', title='Angular speed energy profiles', legend=False
        )
        ax_3.legend(loc='upper right', prop=fontP)  # To add it to the upper right location manually

        # Plot the vibrations heatmap
        ax_5.imshow(
            spectrogram_data,
            norm=matplotlib.colors.LogNorm(),
            cmap='inferno',
            aspect='auto',
            extent=[all_speeds[0], all_speeds[-1], all_angles[0], all_angles[-1]],
            origin='lower',
            interpolation='antialiased',
        )

        # Add vibrations peaks lines in the spectrogram to get hint from peaks found in the first graph
        if vibration_peaks is not None and len(vibration_peaks) > 0:
            for idx, peak in enumerate(vibration_peaks):
                ax_5.axvline(all_speeds[peak], color='cyan', linewidth=0.75)
                ax_5.annotate(
                    f'Peak {idx + 1} ({peak:.1f} Hz)',
                    (all_speeds[peak], all_angles[-1] * 0.9),
                    textcoords='data',
                    color='cyan',
                    rotation=90,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right',
                )
        self.configure_axes(ax_5, xlabel='Speed (mm/s)', ylabel='Angle (deg)', title='Vibrations heatmap', grid=False)

        # Plot the motor profiles
        ax_6.plot(target_freqs, global_motor_profile, label='Combined', color=self.KLIPPAIN_COLORS['purple'], zorder=5)
        max_value = global_motor_profile.max()
        for angle in main_angles:
            profile_max = motor_profiles[angle].max()
            if profile_max > max_value:
                max_value = profile_max
            label = f'{angle_settings.get(angle, (f"{angle} deg",))[0]} ({angle} deg)'
            ax_6.plot(target_freqs, motor_profiles[angle], linestyle='--', label=label, zorder=2)
        ax_6.set_xlim([0, max_freq])
        ax_6.set_ylim([0, max_value * 1.1])

        # Then add the motor resonance peak to the graph
        ax_6.plot(target_freqs[motor_res_idx], global_motor_profile[motor_res_idx], 'x', color='black', markersize=10)
        ax_6.annotate(
            'R',
            (target_freqs[motor_res_idx], global_motor_profile[motor_res_idx]),
            textcoords='offset points',
            xytext=(15, 5),
            ha='right',
            fontsize=14,
            color=self.KLIPPAIN_COLORS['red_pink'],
            weight='bold',
        )

        ax_6_2 = ax_6.twinx()
        ax_6_2.yaxis.set_visible(False)
        ax_6_2.plot([], [], ' ', label=f'Motor resonant frequency (ω0): {motor_fr:.1f}Hz')
        if motor_zeta is not None:
            ax_6_2.plot([], [], ' ', label=f'Motor damping ratio (ζ): {motor_zeta:.3f}')
        else:
            ax_6_2.plot([], [], ' ', label='No damping ratio computed')

        fontP = self.configure_axes(
            ax_6, xlabel='Frequency (Hz)', ylabel='Energy', title='Motor frequency profile', sci_axes='y', legend=True
        )
        ax_6_2.legend(loc='upper right', prop=fontP)

        return fig
