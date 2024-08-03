# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2022 - 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: belts_graph_creator.py
# Description: Implements the CoreXY/CoreXZ belts calibration script for Shake&Tune,
#              including computation and graphing functions for 3D printer belt paths analysis.


import optparse
import os
from datetime import datetime
from typing import List, NamedTuple, Optional, Tuple

import matplotlib
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.stats import pearsonr

matplotlib.use('Agg')

from ..helpers.common_func import detect_peaks, parse_log, setup_klipper_import
from ..helpers.console_output import ConsoleOutput
from ..shaketune_config import ShakeTuneConfig
from .graph_creator import GraphCreator

ALPHABET = (
    'αβγδεζηθικλμνξοπρστυφχψω'  # For paired peak names (using the Greek alphabet to avoid confusion with belt names)
)

PEAKS_DETECTION_THRESHOLD = 0.1  # Threshold to detect peaks in the PSD signal (10% of max)
DC_MAX_PEAKS = 2  # Maximum ideal number of peaks
DC_MAX_UNPAIRED_PEAKS_ALLOWED = 0  # No unpaired peaks are tolerated

KLIPPAIN_COLORS = {
    'purple': '#70088C',
    'orange': '#FF8D32',
    'dark_purple': '#150140',
    'dark_orange': '#F24130',
    'red_pink': '#F2055C',
}


# Define the SignalData type to store the data of a signal (PSD, peaks, etc.)
class SignalData(NamedTuple):
    freqs: np.ndarray
    psd: np.ndarray
    peaks: np.ndarray
    paired_peaks: Optional[List[Tuple[Tuple[int, float, float], Tuple[int, float, float]]]] = None
    unpaired_peaks: Optional[List[int]] = None


# Define the PeakPairingResult type to store the result of the peak pairing function
class PeakPairingResult(NamedTuple):
    paired_peaks: List[Tuple[Tuple[int, float, float], Tuple[int, float, float]]]
    unpaired_peaks1: List[int]
    unpaired_peaks2: List[int]


class BeltsGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config, 'belts comparison')
        self._kinematics: Optional[str] = None
        self._accel_per_hz: Optional[float] = None

    def configure(self, kinematics: Optional[str] = None, accel_per_hz: Optional[float] = None) -> None:
        self._kinematics = kinematics
        self._accel_per_hz = accel_per_hz

    def create_graph(self) -> None:
        lognames = self._move_and_prepare_files(
            glob_pattern='shaketune-belt_*.csv',
            min_files_required=2,
            custom_name_func=lambda f: f.stem.split('_')[1].upper(),
        )
        fig = belts_calibration(
            lognames=[str(path) for path in lognames],
            kinematics=self._kinematics,
            klipperdir=str(self._config.klipper_folder),
            accel_per_hz=self._accel_per_hz,
            st_version=self._version,
        )
        self._save_figure_and_cleanup(fig, lognames)

    def clean_old_files(self, keep_results: int = 3) -> None:
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)
        if len(files) <= keep_results:
            return  # No need to delete any files
        for old_file in files[keep_results:]:
            file_date = '_'.join(old_file.stem.split('_')[1:3])
            for suffix in {'A', 'B'}:
                csv_file = self._folder / f'beltscomparison_{file_date}_{suffix}.csv'
                csv_file.unlink(missing_ok=True)
            old_file.unlink()


######################################################################
# Computation of the PSD graph
######################################################################


# This function create pairs of peaks that are close in frequency on two curves (that are known
# to be resonances points and must be similar on both belts on a CoreXY kinematic)
def pair_peaks(
    peaks1: np.ndarray, freqs1: np.ndarray, psd1: np.ndarray, peaks2: np.ndarray, freqs2: np.ndarray, psd2: np.ndarray
) -> PeakPairingResult:
    # Compute a dynamic detection threshold to filter and pair peaks efficiently
    # even if the signal is very noisy (this get clipped to a maximum of 10Hz diff)
    distances = []
    for p1 in peaks1:
        for p2 in peaks2:
            distances.append(abs(freqs1[p1] - freqs2[p2]))
    distances = np.array(distances)

    median_distance = np.median(distances)
    iqr = np.percentile(distances, 75) - np.percentile(distances, 25)

    threshold = median_distance + 1.5 * iqr
    threshold = min(threshold, 10)

    # Pair the peaks using the dynamic thresold
    paired_peaks = []
    unpaired_peaks1 = list(peaks1)
    unpaired_peaks2 = list(peaks2)

    while unpaired_peaks1 and unpaired_peaks2:
        min_distance = threshold + 1
        pair = None

        for p1 in unpaired_peaks1:
            for p2 in unpaired_peaks2:
                distance = abs(freqs1[p1] - freqs2[p2])
                if distance < min_distance:
                    min_distance = distance
                    pair = (p1, p2)

        if pair is None:  # No more pairs below the threshold
            break

        p1, p2 = pair
        paired_peaks.append(((p1, freqs1[p1], psd1[p1]), (p2, freqs2[p2], psd2[p2])))
        unpaired_peaks1.remove(p1)
        unpaired_peaks2.remove(p2)

    return PeakPairingResult(
        paired_peaks=paired_peaks, unpaired_peaks1=unpaired_peaks1, unpaired_peaks2=unpaired_peaks2
    )


######################################################################
# Computation of the differential spectrogram
######################################################################


def compute_mhi(similarity_factor: float, signal1: SignalData, signal2: SignalData) -> str:
    num_unpaired_peaks = len(signal1.unpaired_peaks) + len(signal2.unpaired_peaks)
    num_paired_peaks = len(signal1.paired_peaks)
    # Combine unpaired peaks from both signals, tagging each peak with its respective signal
    combined_unpaired_peaks = [(peak, signal1) for peak in signal1.unpaired_peaks] + [
        (peak, signal2) for peak in signal2.unpaired_peaks
    ]
    psd_highest_max = max(signal1.psd.max(), signal2.psd.max())

    # Start with the similarity factor directly scaled to a percentage
    mhi = similarity_factor

    # Bonus for ideal number of total peaks (1 or 2)
    if num_paired_peaks >= DC_MAX_PEAKS:
        mhi *= DC_MAX_PEAKS / num_paired_peaks  # Reduce MHI if more than ideal number of peaks

    # Penalty from unpaired peaks weighted by their amplitude relative to the maximum PSD amplitude
    unpaired_peak_penalty = 0
    if num_unpaired_peaks > DC_MAX_UNPAIRED_PEAKS_ALLOWED:
        for peak, signal in combined_unpaired_peaks:
            unpaired_peak_penalty += (signal.psd[peak] / psd_highest_max) * 30
        mhi -= unpaired_peak_penalty

    # Ensure the result lies between 0 and 100 by clipping the computed value
    mhi = np.clip(mhi, 0, 100)

    return mhi_lut(mhi)


# LUT to transform the MHI into a textual value easy to understand for the users of the script
def mhi_lut(mhi: float) -> str:
    ranges = [
        (70, 100, 'Excellent mechanical health'),
        (55, 70, 'Good mechanical health'),
        (45, 55, 'Acceptable mechanical health'),
        (30, 45, 'Potential signs of a mechanical issue'),
        (15, 30, 'Likely a mechanical issue'),
        (0, 15, 'Mechanical issue detected'),
    ]
    mhi = np.clip(mhi, 1, 100)
    return next(
        (message for lower, upper, message in ranges if lower < mhi <= upper),
        'Unknown mechanical health',
    )


######################################################################
# Graphing
######################################################################


def plot_compare_frequency(
    ax: plt.Axes, signal1: SignalData, signal2: SignalData, signal1_belt: str, signal2_belt: str, max_freq: float
) -> None:
    # Plot the two belts PSD signals
    ax.plot(signal1.freqs, signal1.psd, label='Belt ' + signal1_belt, color=KLIPPAIN_COLORS['purple'])
    ax.plot(signal2.freqs, signal2.psd, label='Belt ' + signal2_belt, color=KLIPPAIN_COLORS['orange'])

    psd_highest_max = max(signal1.psd.max(), signal2.psd.max())

    # Trace and annotate the peaks on the graph
    paired_peak_count = 0
    unpaired_peak_count = 0
    offsets_table_data = []

    for _, (peak1, peak2) in enumerate(signal1.paired_peaks):
        label = ALPHABET[paired_peak_count]
        amplitude_offset = abs(((signal2.psd[peak2[0]] - signal1.psd[peak1[0]]) / psd_highest_max) * 100)
        frequency_offset = abs(signal2.freqs[peak2[0]] - signal1.freqs[peak1[0]])
        offsets_table_data.append([f'Peaks {label}', f'{frequency_offset:.1f} Hz', f'{amplitude_offset:.1f} %'])

        ax.plot(signal1.freqs[peak1[0]], signal1.psd[peak1[0]], 'x', color='black')
        ax.plot(signal2.freqs[peak2[0]], signal2.psd[peak2[0]], 'x', color='black')
        ax.plot(
            [signal1.freqs[peak1[0]], signal2.freqs[peak2[0]]],
            [signal1.psd[peak1[0]], signal2.psd[peak2[0]]],
            ':',
            color='gray',
        )

        ax.annotate(
            label + '1',
            (signal1.freqs[peak1[0]], signal1.psd[peak1[0]]),
            textcoords='offset points',
            xytext=(8, 5),
            ha='left',
            fontsize=13,
            color='black',
        )
        ax.annotate(
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
        ax.plot(signal1.freqs[peak], signal1.psd[peak], 'x', color='black')
        ax.annotate(
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
        ax.plot(signal2.freqs[peak], signal2.psd[peak], 'x', color='black')
        ax.annotate(
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
    ax2 = ax.twinx()  # To split the legends in two box
    ax2.yaxis.set_visible(False)
    ax2.plot([], [], ' ', label=f'Number of unpaired peaks: {unpaired_peak_count}')

    # Setting axis parameters, grid and graph title
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([0, max_freq])
    ax.set_ylabel('Power spectral density')
    ax.set_ylim([0, psd_highest_max * 1.1])

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax.set_title(
        'Belts frequency profiles',
        fontsize=14,
        color=KLIPPAIN_COLORS['dark_orange'],
        weight='bold',
    )

    # Print the table of offsets ontop of the graph below the original legend (upper right)
    if len(offsets_table_data) > 0:
        columns = [
            '',
            'Frequency delta',
            'Amplitude delta',
        ]
        offset_table = ax.table(
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

    ax.legend(loc='upper left', prop=fontP)
    ax2.legend(loc='upper right', prop=fontP)

    return


# Compute quantile-quantile plot to compare the two belts
def plot_versus_belts(
    ax: plt.Axes,
    common_freqs: np.ndarray,
    signal1: SignalData,
    signal2: SignalData,
    signal1_belt: str,
    signal2_belt: str,
) -> None:
    ax.set_title('Cross-belts comparison plot', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')

    max_psd = max(np.max(signal1.psd), np.max(signal2.psd))
    ideal_line = np.linspace(0, max_psd * 1.1, 500)
    green_boundary = ideal_line + (0.35 * max_psd * np.exp(-ideal_line / (0.6 * max_psd)))
    ax.fill_betweenx(ideal_line, ideal_line, green_boundary, color='green', alpha=0.15)
    ax.fill_between(ideal_line, ideal_line, green_boundary, color='green', alpha=0.15, label='Good zone')
    ax.plot(
        ideal_line,
        ideal_line,
        '--',
        label='Ideal line',
        color='red',
        linewidth=2,
    )

    ax.plot(signal1.psd, signal2.psd, color='dimgrey', marker='o', markersize=1.5)
    ax.fill_betweenx(signal2.psd, signal1.psd, color=KLIPPAIN_COLORS['red_pink'], alpha=0.1)

    paired_peak_count = 0
    unpaired_peak_count = 0

    for _, (peak1, peak2) in enumerate(signal1.paired_peaks):
        label = ALPHABET[paired_peak_count]
        freq1 = signal1.freqs[peak1[0]]
        freq2 = signal2.freqs[peak2[0]]

        if abs(freq1 - freq2) < 1:
            ax.plot(signal1.psd[peak1[0]], signal2.psd[peak2[0]], marker='o', color='black', markersize=7)
            ax.annotate(
                f'{label}1/{label}2',
                (signal1.psd[peak1[0]], signal2.psd[peak2[0]]),
                textcoords='offset points',
                xytext=(-7, 7),
                fontsize=13,
                color='black',
            )
        else:
            ax.plot(
                signal1.psd[peak2[0]], signal2.psd[peak2[0]], marker='o', color=KLIPPAIN_COLORS['purple'], markersize=7
            )
            ax.plot(
                signal1.psd[peak1[0]], signal2.psd[peak1[0]], marker='o', color=KLIPPAIN_COLORS['orange'], markersize=7
            )
            ax.annotate(
                f'{label}1',
                (signal1.psd[peak1[0]], signal2.psd[peak1[0]]),
                textcoords='offset points',
                xytext=(0, 7),
                fontsize=13,
                color='black',
            )
            ax.annotate(
                f'{label}2',
                (signal1.psd[peak2[0]], signal2.psd[peak2[0]]),
                textcoords='offset points',
                xytext=(0, 7),
                fontsize=13,
                color='black',
            )
        paired_peak_count += 1

    for _, peak_index in enumerate(signal1.unpaired_peaks):
        ax.plot(
            signal1.psd[peak_index], signal2.psd[peak_index], marker='o', color=KLIPPAIN_COLORS['orange'], markersize=7
        )
        ax.annotate(
            str(unpaired_peak_count + 1),
            (signal1.psd[peak_index], signal2.psd[peak_index]),
            textcoords='offset points',
            fontsize=13,
            weight='bold',
            color=KLIPPAIN_COLORS['red_pink'],
            xytext=(0, 7),
        )
        unpaired_peak_count += 1

    for _, peak_index in enumerate(signal2.unpaired_peaks):
        ax.plot(
            signal1.psd[peak_index], signal2.psd[peak_index], marker='o', color=KLIPPAIN_COLORS['purple'], markersize=7
        )
        ax.annotate(
            str(unpaired_peak_count + 1),
            (signal1.psd[peak_index], signal2.psd[peak_index]),
            textcoords='offset points',
            fontsize=13,
            weight='bold',
            color=KLIPPAIN_COLORS['red_pink'],
            xytext=(0, 7),
        )
        unpaired_peak_count += 1

    ax.set_xlabel(f'Belt {signal1_belt}')
    ax.set_ylabel(f'Belt {signal2_belt}')
    ax.set_xlim([0, max_psd * 1.1])
    ax.set_ylim([0, max_psd * 1.1])

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.ticklabel_format(style='scientific', scilimits=(0, 0))
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('medium')
    ax.legend(loc='upper left', prop=fontP)

    return


######################################################################
# Custom tools
######################################################################


# Original Klipper function to get the PSD data of a raw accelerometer signal
def compute_signal_data(data: np.ndarray, common_freqs: np.ndarray, max_freq: float) -> SignalData:
    helper = shaper_calibrate.ShaperCalibrate(printer=None)
    calibration_data = helper.process_accelerometer_data(data)

    freqs = calibration_data.freq_bins[calibration_data.freq_bins <= max_freq]
    psd = calibration_data.get_psd('all')[calibration_data.freq_bins <= max_freq]

    # Re-interpolate the PSD signal to a common frequency range to be able to plot them one against the other
    interp_psd = np.interp(common_freqs, freqs, psd)

    _, peaks, _ = detect_peaks(
        interp_psd, common_freqs, PEAKS_DETECTION_THRESHOLD * interp_psd.max(), window_size=20, vicinity=15
    )

    return SignalData(freqs=common_freqs, psd=interp_psd, peaks=peaks)


######################################################################
# Startup and main routines
######################################################################


def belts_calibration(
    lognames: List[str],
    kinematics: Optional[str],
    klipperdir: str = '~/klipper',
    max_freq: float = 200.0,
    accel_per_hz: Optional[float] = None,
    st_version: str = 'unknown',
) -> plt.Figure:
    global shaper_calibrate
    shaper_calibrate = setup_klipper_import(klipperdir)

    # Parse data from the log files while ignoring CSV in the wrong format
    datas = [data for data in (parse_log(fn) for fn in lognames) if data is not None]
    if len(datas) != 2:
        raise ValueError('Incorrect number of .csv files used (this function needs exactly two files to compare them)!')

    # Get the belts name for the legend to avoid putting the full file name
    belt_info = {'A': ' (axis 1,-1)', 'B': ' (axis 1, 1)'}
    signal1_belt = (lognames[0].split('/')[-1]).split('_')[-1][0]
    signal2_belt = (lognames[1].split('/')[-1]).split('_')[-1][0]
    signal1_belt += belt_info.get(signal1_belt, '')
    signal2_belt += belt_info.get(signal2_belt, '')

    # Compute calibration data for the two datasets with automatic peaks detection
    common_freqs = np.linspace(0, max_freq, 500)
    signal1 = compute_signal_data(datas[0], common_freqs, max_freq)
    signal2 = compute_signal_data(datas[1], common_freqs, max_freq)
    del datas

    # Pair the peaks across the two datasets
    pairing_result = pair_peaks(signal1.peaks, signal1.freqs, signal1.psd, signal2.peaks, signal2.freqs, signal2.psd)
    signal1 = signal1._replace(paired_peaks=pairing_result.paired_peaks, unpaired_peaks=pairing_result.unpaired_peaks1)
    signal2 = signal2._replace(paired_peaks=pairing_result.paired_peaks, unpaired_peaks=pairing_result.unpaired_peaks2)

    # R² proved to be pretty instable to compute the similarity between the two belts
    # So now, we use the Pearson correlation coefficient to compute the similarity
    correlation, _ = pearsonr(signal1.psd, signal2.psd)
    similarity_factor = correlation * 100
    similarity_factor = np.clip(similarity_factor, 0, 100)
    ConsoleOutput.print(f'Belts estimated similarity: {similarity_factor:.1f}%')

    mhi = compute_mhi(similarity_factor, signal1, signal2)
    ConsoleOutput.print(f'[experimental] Mechanical health: {mhi}')

    fig, ((ax1, ax3)) = plt.subplots(
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
        0.060, 0.947, title_line1, ha='left', va='bottom', fontsize=20, color=KLIPPAIN_COLORS['purple'], weight='bold'
    )
    try:
        filename = lognames[0].split('/')[-1]
        dt = datetime.strptime(f"{filename.split('_')[1]} {filename.split('_')[2]}", '%Y%m%d %H%M%S')
        title_line2 = dt.strftime('%x %X')
        if kinematics is not None:
            title_line2 += ' -- ' + kinematics.upper() + ' kinematics'
    except Exception:
        ConsoleOutput.print(f'Warning: Unable to parse the date from the filename ({lognames[0]}, {lognames[1]})')
        title_line2 = lognames[0].split('/')[-1] + ' / ' + lognames[1].split('/')[-1]
    fig.text(0.060, 0.939, title_line2, ha='left', va='top', fontsize=16, color=KLIPPAIN_COLORS['dark_purple'])

    # We add the estimated similarity and the MHI value to the title only if the kinematics is CoreXY
    # as it make no sense to compute these values for other kinematics that doesn't have paired belts
    if kinematics in {'corexy', 'corexz'}:
        title_line3 = f'| Estimated similarity: {similarity_factor:.1f}%'
        title_line4 = f'| {mhi} (experimental)'
        fig.text(0.55, 0.985, title_line3, ha='left', va='top', fontsize=14, color=KLIPPAIN_COLORS['dark_purple'])
        fig.text(0.55, 0.950, title_line4, ha='left', va='top', fontsize=14, color=KLIPPAIN_COLORS['dark_purple'])

    # Add the accel_per_hz value to the title
    title_line5 = f'| Accel per Hz used: {accel_per_hz} mm/s²/Hz'
    fig.text(0.551, 0.915, title_line5, ha='left', va='top', fontsize=10, color=KLIPPAIN_COLORS['dark_purple'])

    # Plot the graphs
    plot_compare_frequency(ax1, signal1, signal2, signal1_belt, signal2_belt, max_freq)
    plot_versus_belts(ax3, common_freqs, signal1, signal2, signal1_belt, signal2_belt)

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
    opts.add_option('-f', '--max_freq', type='float', default=200.0, help='maximum frequency to graph')
    opts.add_option('--accel_per_hz', type='float', default=None, help='accel_per_hz used during the measurement')
    opts.add_option(
        '-k', '--klipper_dir', type='string', dest='klipperdir', default='~/klipper', help='main klipper directory'
    )
    opts.add_option(
        '-m',
        '--kinematics',
        type='string',
        dest='kinematics',
        help='machine kinematics configuration',
    )
    options, args = opts.parse_args()
    if len(args) < 1:
        opts.error('Incorrect number of arguments')
    if options.output is None:
        opts.error('You must specify an output file.png to use the script (option -o)')

    fig = belts_calibration(
        args, options.kinematics, options.klipperdir, options.max_freq, options.accel_per_hz, 'unknown'
    )
    fig.savefig(options.output, dpi=150)


if __name__ == '__main__':
    main()
