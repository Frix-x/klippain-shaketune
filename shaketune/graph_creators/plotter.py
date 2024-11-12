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
from scipy.interpolate import interp1d

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
        st_version = data['st_version']

        fig, ((ax_1, ax_3)) = plt.subplots(
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
        ax_3.remove()
        ax_3 = fig.add_subplot(122, projection='3d')

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot the differents graphs
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

        ax_1.set_xlabel('Time (s)')
        ax_1.set_ylabel('Acceleration (mm/s²)')

        ax_1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_1.grid(which='major', color='grey')
        ax_1.grid(which='minor', color='lightgrey')
        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('small')
        ax_1.set_title(
            'Acceleration (gravity offset removed)',
            fontsize=14,
            color=self.KLIPPAIN_COLORS['dark_orange'],
            weight='bold',
        )

        ax_1.legend(loc='upper left', prop=fontP)

        # Add the gravity and noise level to the graph legend
        ax_1_2 = ax_1.twinx()
        ax_1_2.yaxis.set_visible(False)
        ax_1_2.plot([], [], ' ', label=average_noise_intensity_label)
        ax_1_2.plot([], [], ' ', label=f'Measured gravity: {gravity / 1000:0.3f} m/s²')
        ax_1_2.legend(loc='upper right', prop=fontP)

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot the 3D path of the movement
        for i, ((position_x, position_y, position_z), average_direction_vector, angle_error) in enumerate(
            zip(position_data, direction_vectors, angle_errors)
        ):
            ax_3.plot(
                position_x, position_y, position_z, color=self.KLIPPAIN_COLORS['orange'], linestyle=':', linewidth=2
            )
            ax_3.scatter(position_x[0], position_y[0], position_z[0], color=self.KLIPPAIN_COLORS['red_pink'], zorder=10)
            ax_3.text(
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
            ax_3.plot(
                [start_position[0], end_position[0]],
                [start_position[1], end_position[1]],
                [start_position[2], end_position[2]],
                label=f'{["X", "Y", "Z"][i]} angle: {angle_error:0.2f}°',
                color=self.KLIPPAIN_COLORS['purple'],
                linestyle='-',
                linewidth=2,
            )

        ax_3.set_xlabel('X Position (mm)')
        ax_3.set_ylabel('Y Position (mm)')
        ax_3.set_zlabel('Z Position (mm)')

        ax_3.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_3.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_3.grid(which='major', color='grey')
        ax_3.grid(which='minor', color='lightgrey')
        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('small')
        ax_3.set_title(
            'Estimated movement in 3D space',
            fontsize=14,
            color=self.KLIPPAIN_COLORS['dark_orange'],
            weight='bold',
        )

        ax_3.legend(loc='upper left', prop=fontP)
        # ------------------------------------------------------------------------------------------------------------------------------------------------

        # Add title
        title_line1 = 'AXES MAP CALIBRATION TOOL'
        fig.text(
            0.060,
            0.947,
            title_line1,
            ha='left',
            va='bottom',
            fontsize=20,
            color=self.KLIPPAIN_COLORS['purple'],
            weight='bold',
        )
        try:
            filename = measurements[0]['name']
            dt = datetime.strptime(f"{filename.split('_')[2]} {filename.split('_')[3]}", '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X')
            if self.accel is not None:
                title_line2 += f' -- at {self.accel:0.0f} mm/s²'
        except Exception:
            title_line2 = measurements[0]['name'] + ' ...'
        fig.text(0.060, 0.939, title_line2, ha='left', va='top', fontsize=16, color=self.KLIPPAIN_COLORS['dark_purple'])

        title_line3 = f'| Detected axes_map: {formatted_direction_vector}'
        fig.text(0.50, 0.985, title_line3, ha='left', va='top', fontsize=16, color=self.KLIPPAIN_COLORS['dark_purple'])

        # Adding logo
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, 'klippain.png')
        ax_logo = fig.add_axes([0.001, 0.894, 0.105, 0.105], anchor='NW')
        ax_logo.imshow(plt.imread(image_path))
        ax_logo.axis('off')

        if st_version != 'unknown':
            fig.text(
                0.995, 0.980, st_version, ha='right', va='bottom', fontsize=8, color=self.KLIPPAIN_COLORS['purple']
            )

        return fig

    def plot_static_frequency_graph(self, data):
        # Extract data from computation_result
        freq = data['freq']
        duration = data['duration']
        accel_per_hz = data['accel_per_hz']
        st_version = data['st_version']
        measurements = data['measurements']
        t = data['t']
        bins = data['bins']
        pdata = data['pdata']
        max_freq = data['max_freq']

        fig, ((ax_1, ax_3)) = plt.subplots(
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
        )
        fig.set_size_inches(15, 7)

        title_line1 = 'STATIC FREQUENCY HELPER TOOL'
        fig.text(
            0.060,
            0.947,
            title_line1,
            ha='left',
            va='bottom',
            fontsize=20,
            color=self.KLIPPAIN_COLORS['purple'],
            weight='bold',
        )
        try:
            filename_parts = measurements[0]['name'].split('_')
            dt = datetime.strptime(f'{filename_parts[2]} {filename_parts[3]}', '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X') + ' -- ' + filename_parts[1].upper() + ' axis'
            title_line3 = f'| Maintained frequency: {freq}Hz for {duration}s'
            title_line4 = f'| Accel per Hz used: {accel_per_hz} mm/s²/Hz' if accel_per_hz is not None else ''
        except Exception:
            title_line2 = measurements[0]['name']
            title_line3 = ''
            title_line4 = ''
        fig.text(0.060, 0.939, title_line2, ha='left', va='top', fontsize=16, color=self.KLIPPAIN_COLORS['dark_purple'])
        fig.text(0.55, 0.985, title_line3, ha='left', va='top', fontsize=14, color=self.KLIPPAIN_COLORS['dark_purple'])
        fig.text(0.55, 0.950, title_line4, ha='left', va='top', fontsize=11, color=self.KLIPPAIN_COLORS['dark_purple'])

        ax_1.set_title(
            'Time-Frequency Spectrogram', fontsize=14, color=self.KLIPPAIN_COLORS['dark_orange'], weight='bold'
        )

        vmin_value = np.percentile(pdata, self.SPECTROGRAM_LOW_PERCENTILE_FILTER)

        cm = 'inferno'
        norm = matplotlib.colors.LogNorm(vmin=vmin_value)
        ax_1.imshow(
            pdata.T,
            norm=norm,
            cmap=cm,
            aspect='auto',
            extent=[t[0], t[-1], bins[0], bins[-1]],
            origin='lower',
            interpolation='antialiased',
        )

        ax_1.set_xlim([0.0, max_freq])
        ax_1.set_ylabel('Time (s)')
        ax_1.set_xlabel('Frequency (Hz)')

        # Integrate the energy over the frequency bins for each time step and plot this vertically
        ax_3.plot(np.trapz(pdata, t, axis=0), bins, color=self.KLIPPAIN_COLORS['orange'])
        ax_3.set_title('Vibrations', fontsize=14, color=self.KLIPPAIN_COLORS['dark_orange'], weight='bold')
        ax_3.set_xlabel('Cumulative Energy')
        ax_3.set_ylabel('Time (s)')
        ax_3.set_ylim([bins[0], bins[-1]])

        ax_3.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_3.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_3.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        ax_3.grid(which='major', color='grey')
        ax_3.grid(which='minor', color='lightgrey')

        # Adding logo
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, 'klippain.png')
        ax_logo = fig.add_axes([0.001, 0.894, 0.105, 0.105], anchor='NW')
        ax_logo.imshow(plt.imread(image_path))
        ax_logo.axis('off')

        if st_version != 'unknown':
            fig.text(
                0.995, 0.980, st_version, ha='right', va='bottom', fontsize=8, color=self.KLIPPAIN_COLORS['purple']
            )

        return fig

    def plot_belts_graph(self, data):
        # Extract data from computation_result
        signal1 = data['signal1']
        signal2 = data['signal2']
        similarity_factor = data['similarity_factor']
        mhi = data['mhi']
        signal1_belt = data['signal1_belt']
        signal2_belt = data['signal2_belt']
        kinematics = data['kinematics']
        accel_per_hz = data['accel_per_hz']
        st_version = data['st_version']
        measurements = data['measurements']
        max_freq = data['max_freq']

        # Begin plotting
        fig, ((ax_1, ax_3)) = plt.subplots(
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
        )
        fig.set_size_inches(15, 7)

        # Add title
        title_line1 = 'RELATIVE BELTS CALIBRATION TOOL'
        fig.text(
            0.060,
            0.947,
            title_line1,
            ha='left',
            va='bottom',
            fontsize=20,
            color=self.KLIPPAIN_COLORS['purple'],
            weight='bold',
        )
        try:
            filename = measurements[0]['name']
            dt = datetime.strptime(f"{filename.split('_')[2]} {filename.split('_')[3]}", '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X')
            if kinematics is not None:
                title_line2 += ' -- ' + kinematics.upper() + ' kinematics'
        except Exception:
            title_line2 = measurements[0]['name'] + ' / ' + measurements[1]['name']
        fig.text(0.060, 0.939, title_line2, ha='left', va='top', fontsize=16, color=self.KLIPPAIN_COLORS['dark_purple'])

        # Add similarity and MHI if kinematics is CoreXY
        if kinematics in {'limited_corexy', 'corexy', 'limited_corexz', 'corexz'}:
            title_line3 = f'| Estimated similarity: {similarity_factor:.1f}%'
            title_line4 = f'| {mhi} (experimental)'
            fig.text(
                0.55, 0.985, title_line3, ha='left', va='top', fontsize=14, color=self.KLIPPAIN_COLORS['dark_purple']
            )
            fig.text(
                0.55, 0.950, title_line4, ha='left', va='top', fontsize=14, color=self.KLIPPAIN_COLORS['dark_purple']
            )

        # Add accel_per_hz to the title
        title_line5 = f'| Accel per Hz used: {accel_per_hz} mm/s²/Hz'
        fig.text(0.551, 0.915, title_line5, ha='left', va='top', fontsize=10, color=self.KLIPPAIN_COLORS['dark_purple'])

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot the two belts PSD signals
        ax_1.plot(signal1.freqs, signal1.psd, label='Belt ' + signal1_belt, color=self.KLIPPAIN_COLORS['orange'])
        ax_1.plot(signal2.freqs, signal2.psd, label='Belt ' + signal2_belt, color=self.KLIPPAIN_COLORS['purple'])

        psd_highest_max = max(signal1.psd.max(), signal2.psd.max())

        # Trace and annotate the peaks on the graph
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

        # Add estimated similarity to the graph
        ax_1_2 = ax_1.twinx()  # To split the legends in two box
        ax_1_2.yaxis.set_visible(False)
        ax_1_2.plot([], [], ' ', label=f'Number of unpaired peaks: {unpaired_peak_count}')

        # Setting axis parameters, grid and graph title
        ax_1.set_xlabel('Frequency (Hz)')
        ax_1.set_xlim([0, max_freq])
        ax_1.set_ylabel('Power spectral density')
        ax_1.set_ylim([0, psd_highest_max * 1.1])

        ax_1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_1.grid(which='major', color='grey')
        ax_1.grid(which='minor', color='lightgrey')
        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('small')
        ax_1.set_title(
            'Belts frequency profiles',
            fontsize=14,
            color=self.KLIPPAIN_COLORS['dark_orange'],
            weight='bold',
        )

        # Print the table of offsets ontop of the graph below the original legend (upper right)
        if len(offsets_table_data) > 0:
            columns = [
                '',
                'Frequency delta',
                'Amplitude delta',
            ]
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
            cells = [key for key in offset_table.get_celld().keys()]
            for cell in cells:
                offset_table[cell].set_facecolor('white')
                offset_table[cell].set_alpha(0.6)

        ax_1.legend(loc='upper left', prop=fontP)
        ax_1_2.legend(loc='upper right', prop=fontP)

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        ax_3.set_title(
            'Cross-belts comparison plot', fontsize=14, color=self.KLIPPAIN_COLORS['dark_orange'], weight='bold'
        )

        max_psd = max(np.max(signal1.psd), np.max(signal2.psd))
        ideal_line = np.linspace(0, max_psd * 1.1, 500)
        green_boundary = ideal_line + (0.35 * max_psd * np.exp(-ideal_line / (0.6 * max_psd)))
        ax_3.fill_betweenx(ideal_line, ideal_line, green_boundary, color='green', alpha=0.15)
        ax_3.fill_between(ideal_line, ideal_line, green_boundary, color='green', alpha=0.15, label='Good zone')
        ax_3.plot(
            ideal_line,
            ideal_line,
            '--',
            label='Ideal line',
            color='red',
            linewidth=2,
        )

        ax_3.plot(signal1.psd, signal2.psd, color='dimgrey', marker='o', markersize=1.5)
        ax_3.fill_betweenx(signal2.psd, signal1.psd, color=self.KLIPPAIN_COLORS['red_pink'], alpha=0.1)

        paired_peak_count = 0
        unpaired_peak_count = 0

        for _, (peak1, peak2) in enumerate(signal1.paired_peaks):
            label = self.ALPHABET[paired_peak_count]
            freq1 = signal1.freqs[peak1[0]]
            freq2 = signal2.freqs[peak2[0]]

            if abs(freq1 - freq2) < 1:
                ax_3.plot(signal1.psd[peak1[0]], signal2.psd[peak2[0]], marker='o', color='black', markersize=7)
                ax_3.annotate(
                    f'{label}1/{label}2',
                    (signal1.psd[peak1[0]], signal2.psd[peak2[0]]),
                    textcoords='offset points',
                    xytext=(-7, 7),
                    fontsize=13,
                    color='black',
                )
            else:
                ax_3.plot(
                    signal1.psd[peak2[0]],
                    signal2.psd[peak2[0]],
                    marker='o',
                    color=self.KLIPPAIN_COLORS['purple'],
                    markersize=7,
                )
                ax_3.plot(
                    signal1.psd[peak1[0]],
                    signal2.psd[peak1[0]],
                    marker='o',
                    color=self.KLIPPAIN_COLORS['orange'],
                    markersize=7,
                )
                ax_3.annotate(
                    f'{label}1',
                    (signal1.psd[peak1[0]], signal2.psd[peak1[0]]),
                    textcoords='offset points',
                    xytext=(0, 7),
                    fontsize=13,
                    color='black',
                )
                ax_3.annotate(
                    f'{label}2',
                    (signal1.psd[peak2[0]], signal2.psd[peak2[0]]),
                    textcoords='offset points',
                    xytext=(0, 7),
                    fontsize=13,
                    color='black',
                )
            paired_peak_count += 1

        for _, peak_index in enumerate(signal1.unpaired_peaks):
            ax_3.plot(
                signal1.psd[peak_index],
                signal2.psd[peak_index],
                marker='o',
                color=self.KLIPPAIN_COLORS['orange'],
                markersize=7,
            )
            ax_3.annotate(
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
            ax_3.plot(
                signal1.psd[peak_index],
                signal2.psd[peak_index],
                marker='o',
                color=self.KLIPPAIN_COLORS['purple'],
                markersize=7,
            )
            ax_3.annotate(
                str(unpaired_peak_count + 1),
                (signal1.psd[peak_index], signal2.psd[peak_index]),
                textcoords='offset points',
                fontsize=13,
                weight='bold',
                color=self.KLIPPAIN_COLORS['red_pink'],
                xytext=(0, 7),
            )
            unpaired_peak_count += 1

        ax_3.set_xlabel(f'Belt {signal1_belt}')
        ax_3.set_ylabel(f'Belt {signal2_belt}')
        ax_3.set_xlim([0, max_psd * 1.1])
        ax_3.set_ylim([0, max_psd * 1.1])

        ax_3.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_3.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_3.ticklabel_format(style='scientific', scilimits=(0, 0))
        ax_3.grid(which='major', color='grey')
        ax_3.grid(which='minor', color='lightgrey')

        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('medium')
        ax_3.legend(loc='upper left', prop=fontP)

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Adding logo
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, 'klippain.png')
        ax_logo = fig.add_axes([0.001, 0.894, 0.105, 0.105], anchor='NW')
        ax_logo.imshow(plt.imread(image_path))
        ax_logo.axis('off')

        # Adding version
        if st_version != 'unknown':
            fig.text(
                0.995, 0.980, st_version, ha='right', va='bottom', fontsize=8, color=self.KLIPPAIN_COLORS['purple']
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
        additional_shapers = data['additional_shapers']
        st_version = data['st_version']

        fig, ((ax_1, ax_3), (ax_2, ax_4)) = plt.subplots(
            2,
            2,
            gridspec_kw={
                'height_ratios': [4, 3],
                'width_ratios': [5, 4],
                'bottom': 0.050,
                'top': 0.890,
                'left': 0.048,
                'right': 0.966,
                'hspace': 0.169,
                'wspace': 0.150,
            },
        )
        ax_4.remove()
        fig.set_size_inches(15, 11.6)

        # Add a title with some test info
        title_line1 = 'INPUT SHAPER CALIBRATION TOOL'
        fig.text(
            0.065,
            0.965,
            title_line1,
            ha='left',
            va='bottom',
            fontsize=20,
            color=self.KLIPPAIN_COLORS['purple'],
            weight='bold',
        )
        try:
            filename_parts = measurements[0]['name'].split('_')
            dt = datetime.strptime(f'{filename_parts[2]} {filename_parts[3]}', '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X') + ' -- ' + filename_parts[1].upper() + ' axis'
            if compat:
                title_line3 = '| Older Klipper version detected, damping ratio'
                title_line4 = '| and SCV are not used for filter recommendations!'
                title_line5 = (
                    f'| Accel per Hz used: {self.accel_per_hz} mm/s²/Hz' if self.accel_per_hz is not None else ''
                )
            else:
                max_smoothing_string = (
                    f'maximum ({max_smoothing_computed:0.3f})'
                    if self.max_smoothing is None
                    else f'{self.max_smoothing:0.3f}'
                )
                title_line3 = f'| Square corner velocity: {self.scv} mm/s'
                title_line4 = f'| Allowed smoothing: {max_smoothing_string}'
                title_line5 = (
                    f'| Accel per Hz used: {self.accel_per_hz} mm/s²/Hz' if self.accel_per_hz is not None else ''
                )
        except Exception:
            title_line2 = measurements[0]['name']
            title_line3 = ''
            title_line4 = ''
            title_line5 = ''
        fig.text(0.065, 0.957, title_line2, ha='left', va='top', fontsize=16, color=self.KLIPPAIN_COLORS['dark_purple'])
        fig.text(0.50, 0.990, title_line3, ha='left', va='top', fontsize=14, color=self.KLIPPAIN_COLORS['dark_purple'])
        fig.text(0.50, 0.968, title_line4, ha='left', va='top', fontsize=14, color=self.KLIPPAIN_COLORS['dark_purple'])
        fig.text(0.501, 0.945, title_line5, ha='left', va='top', fontsize=10, color=self.KLIPPAIN_COLORS['dark_purple'])

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        freqs = calibration_data.freqs
        psd = calibration_data.psd_sum
        px = calibration_data.psd_x
        py = calibration_data.psd_y
        pz = calibration_data.psd_z

        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('x-small')

        ax_1.set_xlabel('Frequency (Hz)')
        ax_1.set_xlim([0, max_freq])
        ax_1.set_ylabel('Power spectral density')
        ax_1.set_ylim([0, psd.max() + psd.max() * 0.05])

        ax_1.plot(freqs, psd, label='X+Y+Z', color='purple', zorder=5)
        ax_1.plot(freqs, px, label='X', color='red')
        ax_1.plot(freqs, py, label='Y', color='green')
        ax_1.plot(freqs, pz, label='Z', color='blue')

        ax_1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
        ax_1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_1.grid(which='major', color='grey')
        ax_1.grid(which='minor', color='lightgrey')

        ax_1_2 = ax_1.twinx()
        ax_1_2.yaxis.set_visible(False)

        for shaper in shapers:
            ax_1_2.plot(freqs, shaper.vals, label=shaper.name.upper(), linestyle='dotted')

        # Draw the shappers curves and add their specific parameters in the legend
        for shaper in shaper_table_data['shapers']:
            if shaper['type'] == shaper_choices[0]:
                ax_1.plot(freqs, psd * shaper['vals'], label=f'With {shaper_choices[0]} applied', color='cyan')
            if len(shaper_choices) == 2 and shaper['type'] == shaper_choices[1]:
                ax_1.plot(freqs, psd * shaper['vals'], label=f'With {shaper_choices[1]} applied', color='lime')

        # Draw detected peaks and their respective labels
        ax_1.plot(peaks_freqs, psd[peaks], 'x', color='black', markersize=8)
        for idx, peak in enumerate(peaks):
            if psd[peak] > peaks_threshold[1]:
                fontcolor = 'red'
                fontweight = 'bold'
            else:
                fontcolor = 'black'
                fontweight = 'normal'
            ax_1.annotate(
                f'{idx+1}',
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

        # Add the main resonant frequency and damping ratio of the axis to the graph title
        ax_1.set_title(
            f'Axis Frequency Profile (ω0={fr:.1f}Hz, ζ={zeta:.3f})',
            fontsize=14,
            color=self.KLIPPAIN_COLORS['dark_orange'],
            weight='bold',
        )
        ax_1.legend(loc='upper left', prop=fontP)
        ax_1_2.legend(loc='upper right', prop=fontP)

        # ------------------------------------------------------------------------------------------------------------------
        # Plot a time-frequency spectrogram to see how the system respond over time during the
        # resonnance test. This can highlight hidden spots from the standard PSD graph from other harmonics
        ax_2.set_title(
            'Time-Frequency Spectrogram', fontsize=14, color=self.KLIPPAIN_COLORS['dark_orange'], weight='bold'
        )

        # We need to normalize the data to get a proper signal on the spectrogram
        # However, while using "LogNorm" provide too much background noise, using
        # "Normalize" make only the resonnance appearing and hide interesting elements
        # So we need to filter out the lower part of the data (ie. find the proper vmin for LogNorm)
        vmin_value = np.percentile(pdata, self.SPECTROGRAM_LOW_PERCENTILE_FILTER)
        cm = 'inferno'
        norm = matplotlib.colors.LogNorm(vmin=vmin_value)

        # Draw the spectrogram using imgshow that is better suited here than pcolormesh since its result is already rasterized and
        # we doesn't need to keep vector graphics when saving to a final .png file. Using it also allow to
        # save ~150-200MB of RAM during the "fig.savefig" operation.
        ax_2.imshow(
            pdata.T,
            norm=norm,
            cmap=cm,
            aspect='auto',
            extent=[t[0], t[-1], bins[0], bins[-1]],
            origin='lower',
            interpolation='antialiased',
        )

        ax_2.set_xlim([0.0, max_freq])
        ax_2.set_ylabel('Time (s)')
        ax_2.set_xlabel('Frequency (Hz)')

        # Add peaks lines in the spectrogram to get hint from peaks found in the first graph
        for idx, peak in enumerate(peaks):
            ax_2.axvline(peak, color='cyan', linestyle='dotted', linewidth=1)
            ax_2.annotate(
                f'Peak {idx+1}',
                (peak, bins[-1] * 0.9),
                textcoords='data',
                color='cyan',
                rotation=90,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
            )

        # ------------------------------------------------------------------------------------------------------------------
        ax_3.set_title(
            'Filters performances over acceleration',
            fontsize=14,
            color=self.KLIPPAIN_COLORS['dark_orange'],
            weight='bold',
        )
        ax_3.set_xlabel('Max Acceleration')
        ax_3.set_ylabel('Remaining Vibrations (%)')

        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('x-small')

        ax_3.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1000))
        ax_3.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_3.grid(which='major', color='grey')
        ax_3.grid(which='minor', color='lightgrey')

        shaper_data = {}

        # Extract data from additional_shapers first
        for _, shapers in additional_shapers.items():
            for shaper in shapers:
                shaper_type = shaper.name.upper()
                if shaper_type not in shaper_data:
                    shaper_data[shaper_type] = []
                shaper_data[shaper_type].append(
                    {
                        'max_accel': shaper.max_accel,
                        'vibrs': shaper.vibrs * 100.0,
                    }
                )

        # Extract data from shaper_table_data and insert into shaper_data
        max_shaper_vibrations = 0
        for shaper in shaper_table_data['shapers']:
            shaper_type = shaper['type']
            if shaper_type not in shaper_data:
                shaper_data[shaper_type] = []
            max_shaper_vibrations = max(max_shaper_vibrations, float(shaper['vibrations']) * 100.0)
            shaper_data[shaper_type].append(
                {
                    'max_accel': float(shaper['max_accel']),
                    'vibrs': float(shaper['vibrations']) * 100.0,
                }
            )

        # Calculate the maximum `max_accel` for points below the thresholds to get a good plot with
        # continuous lines and a zoom on the graph to show details at low vibrations
        min_accel_limit = 99999
        max_accel_limit = 0
        max_accel_limit_zoom = 0
        for data in shaper_data.values():
            min_accel_limit = min(min_accel_limit, min(d['max_accel'] for d in data))
            max_accel_limit = max(
                max_accel_limit, max(d['max_accel'] for d in data if d['vibrs'] <= self.MAX_VIBRATIONS_PLOTTED)
            )
            max_accel_limit_zoom = max(
                max_accel_limit_zoom,
                max(
                    d['max_accel']
                    for d in data
                    if d['vibrs'] <= max_shaper_vibrations * self.MAX_VIBRATIONS_PLOTTED_ZOOM
                ),
            )

        # Add a zoom axes on the graph to show details at low vibrations
        zoomed_window = np.clip(max_shaper_vibrations * self.MAX_VIBRATIONS_PLOTTED_ZOOM, 0, 20)
        axins = ax_3.inset_axes(
            [0.575, 0.125, 0.40, 0.45],
            xlim=(min_accel_limit * 0.95, max_accel_limit_zoom * 1.1),
            ylim=(-0.5, zoomed_window),
        )
        ax_3.indicate_inset_zoom(axins, edgecolor=self.KLIPPAIN_COLORS['purple'], linewidth=3)
        axins.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
        axins.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axins.grid(which='major', color='grey')
        axins.grid(which='minor', color='lightgrey')

        # Draw the green zone on both axes to highlight the low vibrations zone
        number_of_interpolated_points = 100
        x_fill = np.linspace(min_accel_limit * 0.95, max_accel_limit * 1.1, number_of_interpolated_points)
        y_fill = np.full_like(x_fill, 5.0)
        ax_3.axhline(y=5.0, color='black', linestyle='--', linewidth=0.5)
        ax_3.fill_between(x_fill, -0.5, y_fill, color='green', alpha=0.15)
        if zoomed_window > 5.0:
            axins.axhline(y=5.0, color='black', linestyle='--', linewidth=0.5)
            axins.fill_between(x_fill, -0.5, y_fill, color='green', alpha=0.15)

        # Plot each shaper remaining vibrations response over acceleration
        max_vibrations = 0
        for _, (shaper_type, data) in enumerate(shaper_data.items()):
            max_accel_values = np.array([d['max_accel'] for d in data])
            vibrs_values = np.array([d['vibrs'] for d in data])

            # remove duplicate values in max_accel_values and delete the corresponding vibrs_values
            # and interpolate the curves to get them smoother with more datapoints
            unique_max_accel_values, unique_indices = np.unique(max_accel_values, return_index=True)
            max_accel_values = unique_max_accel_values
            vibrs_values = vibrs_values[unique_indices]
            interp_func = interp1d(max_accel_values, vibrs_values, kind='cubic')
            max_accel_fine = np.linspace(max_accel_values.min(), max_accel_values.max(), number_of_interpolated_points)
            vibrs_fine = interp_func(max_accel_fine)

            ax_3.plot(max_accel_fine, vibrs_fine, label=f'{shaper_type}', zorder=10)
            axins.plot(max_accel_fine, vibrs_fine, label=f'{shaper_type}', zorder=15)
            max_vibrations = max(max_vibrations, max(vibrs_fine))

        ax_3.set_xlim([min_accel_limit * 0.95, max_accel_limit * 1.1])
        ax_3.set_ylim([-0.5, np.clip(max_vibrations * 1.05, 50, self.MAX_VIBRATIONS_PLOTTED)])
        ax_3.legend(loc='best', prop=fontP)

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Print the table of offsets ontop of the graph below the original legend (upper right)
        columns = ['Type', 'Frequency', 'Vibrations', 'Smoothing', 'Max Accel']
        table_data = []

        for shaper_info in shaper_table_data['shapers']:
            row = [
                f'{shaper_info["type"].upper()}',
                f'{shaper_info["frequency"]:.1f} Hz',
                f'{shaper_info["vibrations"] * 100:.1f} %',
                f'{shaper_info["smoothing"]:.3f}',
                f'{round(shaper_info["max_accel"] / 10) * 10:.0f}',
            ]
            table_data.append(row)
        table = plt.table(cellText=table_data, colLabels=columns, bbox=[1.130, -0.4, 0.803, 0.25], cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2, 3, 4])
        table.set_zorder(100)

        # Add the recommendations and damping ratio using fig.text
        fig.text(
            0.585,
            0.235,
            f'Estimated damping ratio (ζ): {shaper_table_data["damping_ratio"]:.3f}',
            fontsize=14,
            color=self.KLIPPAIN_COLORS['purple'],
        )
        if len(shaper_table_data['recommendations']) == 1:
            fig.text(
                0.585,
                0.200,
                shaper_table_data['recommendations'][0],
                fontsize=14,
                color=self.KLIPPAIN_COLORS['red_pink'],
            )
        elif len(shaper_table_data['recommendations']) == 2:
            fig.text(
                0.585,
                0.200,
                shaper_table_data['recommendations'][0],
                fontsize=14,
                color=self.KLIPPAIN_COLORS['red_pink'],
            )
            fig.text(
                0.585,
                0.175,
                shaper_table_data['recommendations'][1],
                fontsize=14,
                color=self.KLIPPAIN_COLORS['red_pink'],
            )
        # ------------------------------------------------------------------------------------------------------------------------------------------------

        # Adding a small Klippain logo to the top left corner of the figure
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, 'klippain.png')
        ax_logo = fig.add_axes([0.001, 0.924, 0.075, 0.075], anchor='NW')
        ax_logo.imshow(plt.imread(image_path))
        ax_logo.axis('off')

        # Adding Shake&Tune version in the top right corner
        if st_version != 'unknown':
            fig.text(
                0.995, 0.985, st_version, ha='right', va='bottom', fontsize=8, color=self.KLIPPAIN_COLORS['purple']
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

        fig, ((ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6)) = plt.subplots(
            2,
            3,
            gridspec_kw={
                'height_ratios': [1, 1],
                'width_ratios': [4, 8, 6],
                'bottom': 0.050,
                'top': 0.890,
                'left': 0.040,
                'right': 0.985,
                'hspace': 0.166,
                'wspace': 0.138,
            },
        )

        # Transform ax_3 and ax_4 to polar plots
        ax_1.remove()
        ax_1 = fig.add_subplot(2, 3, 1, projection='polar')
        ax_4.remove()
        ax_4 = fig.add_subplot(2, 3, 4, projection='polar')

        # Set the global .png figure size
        fig.set_size_inches(20, 11.5)

        # Add title
        title_line1 = 'MACHINE VIBRATIONS ANALYSIS TOOL'
        fig.text(
            0.060,
            0.965,
            title_line1,
            ha='left',
            va='bottom',
            fontsize=20,
            color=self.KLIPPAIN_COLORS['purple'],
            weight='bold',
        )
        try:
            filename_parts = measurements[0]['name'].split('_')
            dt = datetime.strptime(f"{filename_parts[4]} {filename_parts[5].split('-')[0]}", '%Y%m%d %H%M%S')
            title_line2 = dt.strftime('%x %X')
            if accel is not None:
                title_line2 += ' at ' + str(accel) + ' mm/s² -- ' + kinematics.upper() + ' kinematics'
        except Exception:
            title_line2 = measurements[0]['name']
        fig.text(0.060, 0.957, title_line2, ha='left', va='top', fontsize=16, color=self.KLIPPAIN_COLORS['dark_purple'])

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot the motors infos to the top of the graph if they are detected / specified (not mandatory for CLI mode)
        if motors is not None and len(motors) == 2:
            motor_details = [(motors[0], 'X motor'), (motors[1], 'Y motor')]

            distance = 0.12
            if motors[0].get_config('autotune_enabled'):
                distance = 0.27
                config_blocks = [
                    f"| {lbl}: {mot.get_config('motor').upper()} on {mot.get_config('tmc').upper()} @ {mot.get_config('voltage'):0.1f}V {mot.get_config('run_current'):0.2f}A - {mot.get_config('microsteps')}usteps"
                    for mot, lbl in motor_details
                ]
                config_blocks.append(
                    f'| TMC Autotune enabled (PWM freq target: X={int(motors[0].get_config("pwm_freq_target")/1000)}kHz / Y={int(motors[1].get_config("pwm_freq_target")/1000)}kHz)'
                )
            else:
                config_blocks = [
                    f"| {lbl}: {mot.get_config('tmc').upper()} @ {mot.get_config('run_current'):0.2f}A - {mot.get_config('microsteps')}usteps"
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

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot the angle profile in polar coordinates
        angles_radians = np.deg2rad(all_angles)

        ax_1.set_title(
            'Polar angle energy profile', fontsize=14, color=self.KLIPPAIN_COLORS['dark_orange'], weight='bold'
        )
        ax_1.set_theta_zero_location('E')
        ax_1.set_theta_direction(1)

        ax_1.plot(angles_radians, all_angles_energy, color=self.KLIPPAIN_COLORS['purple'], zorder=5)
        ax_1.fill(angles_radians, all_angles_energy, color=self.KLIPPAIN_COLORS['purple'], alpha=0.3)
        ax_1.set_xlim([0, np.deg2rad(360)])
        ymax = all_angles_energy.max() * 1.05
        ax_1.set_ylim([0, ymax])
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

        ax_1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_1.grid(which='major', color='grey')
        ax_1.grid(which='minor', color='lightgrey')

        # Polar plot doesn't follow the gridspec margin, so we adjust it manually here
        pos = ax_1.get_position()
        new_pos = [pos.x0 - 0.01, pos.y0 - 0.01, pos.width, pos.height]
        ax_1.set_position(new_pos)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Plot the vibration spectrogram in polar coordinates

        # Assuming speeds defines the radial distance from the center, we need to create a meshgrid
        # for both angles and speeds to map the spectrogram data onto a polar plot correctly
        radius, theta = np.meshgrid(all_speeds, angles_radians)

        ax_4.set_title(
            'Polar vibrations heatmap',
            fontsize=14,
            color=self.KLIPPAIN_COLORS['dark_orange'],
            weight='bold',
            va='bottom',
        )
        ax_4.set_theta_zero_location('E')
        ax_4.set_theta_direction(1)

        ax_4.pcolormesh(
            theta, radius, spectrogram_data, norm=matplotlib.colors.LogNorm(), cmap='inferno', shading='auto'
        )
        ax_4.set_thetagrids([theta * 15 for theta in range(360 // 15)])
        ax_4.tick_params(axis='y', which='both', colors='white', labelsize='medium')
        ax_4.set_ylim([0, max(all_speeds)])

        # Polar plot doesn't follow the gridspec margin, so we adjust it manually here
        pos = ax_4.get_position()
        new_pos = [pos.x0 - 0.01, pos.y0 - 0.01, pos.width, pos.height]
        ax_4.set_position(new_pos)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Plot global speed profile
        ax_2.set_title(
            'Global speed energy profile', fontsize=14, color=self.KLIPPAIN_COLORS['dark_orange'], weight='bold'
        )
        ax_2.set_xlabel('Speed (mm/s)')
        ax_2.set_ylabel('Energy')
        ax_2_2 = ax_2.twinx()
        ax_2_2.yaxis.set_visible(False)

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
        ax_2_2.plot(
            all_speeds,
            vibration_metric,
            label=f'Vibration metric ({num_peaks} bad peaks)',
            color=self.KLIPPAIN_COLORS['red_pink'],
            zorder=5,
        )

        ax_2.set_xlim([all_speeds.min(), all_speeds.max()])
        ax_2.set_ylim([0, sp_max_energy.max() * 1.15])

        y2min = -(vibration_metric.max() * 0.025)
        y2max = vibration_metric.max() * 1.07
        ax_2_2.set_ylim([y2min, y2max])

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
                    f'{idx+1}',
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
            # ax_2_2.axvline(all_speeds[start], color=self.KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5, zorder=8)
            # ax_2_2.axvline(all_speeds[end], color=self.KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5, zorder=8)
            ax_2_2.fill_between(
                all_speeds[start:end],
                y2min,
                vibration_metric[start:end],
                color='green',
                alpha=0.2,
                label=f'Zone {idx+1}: {all_speeds[start]:.1f} to {all_speeds[end]:.1f} mm/s',
            )

        ax_2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_2.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_2.grid(which='major', color='grey')
        ax_2.grid(which='minor', color='lightgrey')

        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('small')
        ax_2.legend(loc='upper left', prop=fontP)
        ax_2_2.legend(loc='upper right', prop=fontP)

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot the angular speed profile
        ax_3.set_title(
            'Angular speed energy profiles', fontsize=14, color=self.KLIPPAIN_COLORS['dark_orange'], weight='bold'
        )
        ax_3.set_xlabel('Speed (mm/s)')
        ax_3.set_ylabel('Energy')

        # Define mappings for labels and colors to simplify plotting commands
        angle_settings = {
            0: ('X (0 deg)', 'purple', 10),
            90: ('Y (90 deg)', 'dark_purple', 5),
            45: ('A (45 deg)' if kinematics in {'corexy', 'limited_corexy'} else '45 deg', 'orange', 10),
            135: ('B (135 deg)' if kinematics in {'corexy', 'limited_corexy'} else '135 deg', 'dark_orange', 5),
        }

        # Plot each angle using settings from the dictionary
        for angle, (label, color, zorder) in angle_settings.items():
            idx = np.searchsorted(all_angles, angle, side='left')
            ax_3.plot(all_speeds, spectrogram_data[idx], label=label, color=self.KLIPPAIN_COLORS[color], zorder=zorder)

        ax_3.set_xlim([all_speeds.min(), all_speeds.max()])
        max_value = max(spectrogram_data[angle].max() for angle in {0, 45, 90, 135})
        ax_3.set_ylim([0, max_value * 1.1])

        ax_3.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_3.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_3.grid(which='major', color='grey')
        ax_3.grid(which='minor', color='lightgrey')

        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('small')
        ax_3.legend(loc='upper right', prop=fontP)

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot the vibration spectrogram
        ax_5.set_title('Vibrations heatmap', fontsize=14, color=self.KLIPPAIN_COLORS['dark_orange'], weight='bold')
        ax_5.set_xlabel('Speed (mm/s)')
        ax_5.set_ylabel('Angle (deg)')

        ax_5.imshow(
            spectrogram_data,
            norm=matplotlib.colors.LogNorm(),
            cmap='inferno',
            aspect='auto',
            extent=[all_speeds[0], all_speeds[-1], all_angles[0], all_angles[-1]],
            origin='lower',
            interpolation='antialiased',
        )

        # Add peaks lines in the spectrogram to get hint from peaks found in the first graph
        if vibration_peaks is not None and len(vibration_peaks) > 0:
            for idx, peak in enumerate(vibration_peaks):
                ax_5.axvline(all_speeds[peak], color='cyan', linewidth=0.75)
                ax_5.annotate(
                    f'Peak {idx+1}',
                    (all_speeds[peak], all_angles[-1] * 0.9),
                    textcoords='data',
                    color='cyan',
                    rotation=90,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right',
                )

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot the motor profiles
        ax_6.set_title('Motor frequency profile', fontsize=14, color=self.KLIPPAIN_COLORS['dark_orange'], weight='bold')
        ax_6.set_ylabel('Energy')
        ax_6.set_xlabel('Frequency (Hz)')

        ax_6_2 = ax_6.twinx()
        ax_6_2.yaxis.set_visible(False)

        # Global weighted average motor profile
        ax_6.plot(target_freqs, global_motor_profile, label='Combined', color=self.KLIPPAIN_COLORS['purple'], zorder=5)
        max_value = global_motor_profile.max()

        # Mapping of angles to axis names
        angle_settings = {0: 'X', 90: 'Y', 45: 'A', 135: 'B'}

        # And then plot the motor profiles at each measured angles
        for angle in main_angles:
            profile_max = motor_profiles[angle].max()
            if profile_max > max_value:
                max_value = profile_max
            label = f'{angle_settings[angle]} ({angle} deg)' if angle in angle_settings else f'{angle} deg'
            ax_6.plot(target_freqs, motor_profiles[angle], linestyle='--', label=label, zorder=2)

        ax_6.set_xlim([0, max_freq])
        ax_6.set_ylim([0, max_value * 1.1])
        ax_6.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

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

        ax_6_2.plot([], [], ' ', label=f'Motor resonant frequency (ω0): {motor_fr:.1f}Hz')
        if motor_zeta is not None:
            ax_6_2.plot([], [], ' ', label=f'Motor damping ratio (ζ): {motor_zeta:.3f}')
        else:
            ax_6_2.plot([], [], ' ', label='No damping ratio computed')

        ax_6.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_6.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_6.grid(which='major', color='grey')
        ax_6.grid(which='minor', color='lightgrey')

        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('small')
        ax_6.legend(loc='upper left', prop=fontP)
        ax_6_2.legend(loc='upper right', prop=fontP)

        # Adding a small Klippain logo to the top left corner of the figure
        ax_logo = fig.add_axes([0.001, 0.924, 0.075, 0.075], anchor='NW')
        ax_logo.imshow(plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'klippain.png')))
        ax_logo.axis('off')

        # Adding Shake&Tune version in the top right corner
        if st_version != 'unknown':
            fig.text(
                0.995, 0.985, st_version, ha='right', va='bottom', fontsize=8, color=self.KLIPPAIN_COLORS['purple']
            )

        return fig
