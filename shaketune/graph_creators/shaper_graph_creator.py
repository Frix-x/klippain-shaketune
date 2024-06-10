#!/usr/bin/env python3

#################################################
######## INPUT SHAPER CALIBRATION SCRIPT ########
#################################################
# Derived from the calibrate_shaper.py official Klipper script
# Copyright (C) 2020  Dmitry Butyugin <dmbutyugin@google.com>
# Copyright (C) 2020  Kevin O'Connor <kevin@koconnor.net>
# Highly modified and improved by Frix_x#0161 #

import optparse
import os
from datetime import datetime
from typing import List, Optional

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

matplotlib.use('Agg')

from ..helpers.common_func import (
    compute_mechanical_parameters,
    compute_spectrogram,
    detect_peaks,
    parse_log,
    setup_klipper_import,
)
from ..helpers.console_output import ConsoleOutput
from ..shaketune_config import ShakeTuneConfig
from .graph_creator import GraphCreator

PEAKS_DETECTION_THRESHOLD = 0.05
PEAKS_EFFECT_THRESHOLD = 0.12
SPECTROGRAM_LOW_PERCENTILE_FILTER = 5
MAX_VIBRATIONS = 5.0

KLIPPAIN_COLORS = {
    'purple': '#70088C',
    'orange': '#FF8D32',
    'dark_purple': '#150140',
    'dark_orange': '#F24130',
    'red_pink': '#F2055C',
}


class ShaperGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config, 'input shaper')
        self._max_smoothing: Optional[float] = None
        self._scv: Optional[float] = None
        self._accel_per_hz: Optional[float] = None

    def configure(
        self, scv: float, max_smoothing: Optional[float] = None, accel_per_hz: Optional[float] = None
    ) -> None:
        self._scv = scv
        self._max_smoothing = max_smoothing
        self._accel_per_hz = accel_per_hz

    def create_graph(self) -> None:
        if not self._scv:
            raise ValueError('scv must be set to create the input shaper graph!')

        lognames = self._move_and_prepare_files(
            glob_pattern='shaketune-axis_*.csv',
            min_files_required=1,
            custom_name_func=lambda f: f.stem.split('_')[1].upper(),
        )
        fig = shaper_calibration(
            lognames=[str(path) for path in lognames],
            klipperdir=str(self._config.klipper_folder),
            max_smoothing=self._max_smoothing,
            scv=self._scv,
            accel_per_hz=self._accel_per_hz,
            st_version=self._version,
        )
        self._save_figure_and_cleanup(fig, lognames, lognames[0].stem.split('_')[-1])

    def clean_old_files(self, keep_results: int = 3) -> None:
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)
        if len(files) <= 2 * keep_results:
            return  # No need to delete any files
        for old_file in files[2 * keep_results :]:
            csv_file = old_file.with_suffix('.csv')
            csv_file.unlink(missing_ok=True)
            old_file.unlink()


######################################################################
# Computation
######################################################################


# Find the best shaper parameters using Klipper's official algorithm selection with
# a proper precomputed damping ratio (zeta) and using the configured printer SQV value
def calibrate_shaper(datas: List[np.ndarray], max_smoothing: Optional[float], scv: float, max_freq: float):
    helper = shaper_calibrate.ShaperCalibrate(printer=None)
    calibration_data = helper.process_accelerometer_data(datas)
    calibration_data.normalize_to_frequencies()

    fr, zeta, _, _ = compute_mechanical_parameters(calibration_data.psd_sum, calibration_data.freq_bins)

    # If the damping ratio computation fail, we use Klipper default value instead
    if zeta is None:
        zeta = 0.1

    compat = False
    try:
        shaper, all_shapers = helper.find_best_shaper(
            calibration_data,
            shapers=None,
            damping_ratio=zeta,
            scv=scv,
            shaper_freqs=None,
            max_smoothing=max_smoothing,
            test_damping_ratios=None,
            max_freq=max_freq,
            logger=ConsoleOutput.print,
        )
    except TypeError:
        ConsoleOutput.print(
            '[WARNING] You seem to be using an older version of Klipper that is not compatible with all the latest Shake&Tune features!'
        )
        ConsoleOutput.print(
            'Shake&Tune now runs in compatibility mode: be aware that the results may be slightly off, since the real damping ratio cannot be used to create the filter recommendations'
        )
        compat = True
        shaper, all_shapers = helper.find_best_shaper(calibration_data, max_smoothing, ConsoleOutput.print)

    ConsoleOutput.print(
        '\n-> Recommended shaper is %s @ %.1f Hz (when using a square corner velocity of %.1f and a damping ratio of %.3f)'
        % (shaper.name.upper(), shaper.freq, scv, zeta)
    )

    return shaper.name, all_shapers, calibration_data, fr, zeta, compat


######################################################################
# Graphing
######################################################################


def plot_freq_response(
    ax: plt.Axes,
    calibration_data,
    shapers,
    klipper_shaper_choice: str,
    peaks: np.ndarray,
    peaks_freqs: np.ndarray,
    peaks_threshold: List[float],
    fr: float,
    zeta: float,
    max_freq: float,
) -> None:
    freqs = calibration_data.freqs
    psd = calibration_data.psd_sum
    px = calibration_data.psd_x
    py = calibration_data.psd_y
    pz = calibration_data.psd_z

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('x-small')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([0, max_freq])
    ax.set_ylabel('Power spectral density')
    ax.set_ylim([0, psd.max() + psd.max() * 0.05])

    ax.plot(freqs, psd, label='X+Y+Z', color='purple', zorder=5)
    ax.plot(freqs, px, label='X', color='red')
    ax.plot(freqs, py, label='Y', color='green')
    ax.plot(freqs, pz, label='Z', color='blue')

    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    ax2 = ax.twinx()
    ax2.yaxis.set_visible(False)

    # Draw the shappers curves and add their specific parameters in the legend
    perf_shaper_choice = None
    perf_shaper_vals = None
    perf_shaper_freq = None
    perf_shaper_accel = 0
    for shaper in shapers:
        shaper_max_accel = round(shaper.max_accel / 100.0) * 100.0
        label = '%s (%.1f Hz, vibr=%.1f%%, sm~=%.2f, accel<=%.f)' % (
            shaper.name.upper(),
            shaper.freq,
            shaper.vibrs * 100.0,
            shaper.smoothing,
            shaper_max_accel,
        )
        ax2.plot(freqs, shaper.vals, label=label, linestyle='dotted')

        # Get the Klipper recommended shaper (usually it's a good low vibration compromise)
        if shaper.name == klipper_shaper_choice:
            klipper_shaper_freq = shaper.freq
            klipper_shaper_vals = shaper.vals
            klipper_shaper_accel = shaper_max_accel

        # Find the shaper with the highest accel but with vibrs under MAX_VIBRATIONS as it's
        # a good performance compromise when injecting the SCV and damping ratio in the computation
        if perf_shaper_accel < shaper_max_accel and shaper.vibrs * 100 < MAX_VIBRATIONS:
            perf_shaper_choice = shaper.name
            perf_shaper_accel = shaper_max_accel
            perf_shaper_freq = shaper.freq
            perf_shaper_vals = shaper.vals

    # Recommendations are added to the legend: one is Klipper's original suggestion that is usually good for low vibrations
    # and the other one is the custom "performance" recommendation that looks for a suitable shaper that doesn't have excessive
    # vibrations level but have higher accelerations. If both recommendations are the same shaper, or if no suitable "performance"
    # shaper is found, then only a single line as the "best shaper" recommendation is added to the legend
    if (
        perf_shaper_choice is not None
        and perf_shaper_choice != klipper_shaper_choice
        and perf_shaper_accel >= klipper_shaper_accel
    ):
        ax2.plot(
            [],
            [],
            ' ',
            label='Recommended performance shaper: %s @ %.1f Hz' % (perf_shaper_choice.upper(), perf_shaper_freq),
        )
        ax.plot(
            freqs,
            psd * perf_shaper_vals,
            label='With %s applied' % (perf_shaper_choice.upper()),
            color='cyan',
        )
        ax2.plot(
            [],
            [],
            ' ',
            label='Recommended low vibrations shaper: %s @ %.1f Hz'
            % (klipper_shaper_choice.upper(), klipper_shaper_freq),
        )
        ax.plot(
            freqs, psd * klipper_shaper_vals, label='With %s applied' % (klipper_shaper_choice.upper()), color='lime'
        )
    else:
        ax2.plot(
            [],
            [],
            ' ',
            label='Recommended best shaper: %s @ %.1f Hz' % (klipper_shaper_choice.upper(), klipper_shaper_freq),
        )
        ax.plot(
            freqs,
            psd * klipper_shaper_vals,
            label='With %s applied' % (klipper_shaper_choice.upper()),
            color='cyan',
        )

    # And the estimated damping ratio is finally added at the end of the legend
    ax2.plot([], [], ' ', label='Estimated damping ratio (ζ): %.3f' % (zeta))

    # Draw the detected peaks and name them
    # This also draw the detection threshold and warning threshold (aka "effect zone")
    ax.plot(peaks_freqs, psd[peaks], 'x', color='black', markersize=8)
    for idx, peak in enumerate(peaks):
        if psd[peak] > peaks_threshold[1]:
            fontcolor = 'red'
            fontweight = 'bold'
        else:
            fontcolor = 'black'
            fontweight = 'normal'
        ax.annotate(
            f'{idx+1}',
            (freqs[peak], psd[peak]),
            textcoords='offset points',
            xytext=(8, 5),
            ha='left',
            fontsize=13,
            color=fontcolor,
            weight=fontweight,
        )
    ax.axhline(y=peaks_threshold[0], color='black', linestyle='--', linewidth=0.5)
    ax.axhline(y=peaks_threshold[1], color='black', linestyle='--', linewidth=0.5)
    ax.fill_between(freqs, 0, peaks_threshold[0], color='green', alpha=0.15, label='Relax Region')
    ax.fill_between(freqs, peaks_threshold[0], peaks_threshold[1], color='orange', alpha=0.2, label='Warning Region')

    # Add the main resonant frequency and damping ratio of the axis to the graph title
    ax.set_title(
        'Axis Frequency Profile (ω0=%.1fHz, ζ=%.3f)' % (fr, zeta),
        fontsize=14,
        color=KLIPPAIN_COLORS['dark_orange'],
        weight='bold',
    )
    ax.legend(loc='upper left', prop=fontP)
    ax2.legend(loc='upper right', prop=fontP)

    return


# Plot a time-frequency spectrogram to see how the system respond over time during the
# resonnance test. This can highlight hidden spots from the standard PSD graph from other harmonics
def plot_spectrogram(
    ax: plt.Axes, t: np.ndarray, bins: np.ndarray, pdata: np.ndarray, peaks: np.ndarray, max_freq: float
) -> None:
    ax.set_title('Time-Frequency Spectrogram', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')

    # We need to normalize the data to get a proper signal on the spectrogram
    # However, while using "LogNorm" provide too much background noise, using
    # "Normalize" make only the resonnance appearing and hide interesting elements
    # So we need to filter out the lower part of the data (ie. find the proper vmin for LogNorm)
    vmin_value = np.percentile(pdata, SPECTROGRAM_LOW_PERCENTILE_FILTER)

    # Draw the spectrogram using imgshow that is better suited here than pcolormesh since its result is already rasterized and
    # we doesn't need to keep vector graphics when saving to a final .png file. Using it also allow to
    # save ~150-200MB of RAM during the "fig.savefig" operation.
    cm = 'inferno'
    norm = matplotlib.colors.LogNorm(vmin=vmin_value)
    ax.imshow(
        pdata.T,
        norm=norm,
        cmap=cm,
        aspect='auto',
        extent=[t[0], t[-1], bins[0], bins[-1]],
        origin='lower',
        interpolation='antialiased',
    )

    ax.set_xlim([0.0, max_freq])
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Frequency (Hz)')

    # Add peaks lines in the spectrogram to get hint from peaks found in the first graph
    if peaks is not None:
        for idx, peak in enumerate(peaks):
            ax.axvline(peak, color='cyan', linestyle='dotted', linewidth=1)
            ax.annotate(
                f'Peak {idx+1}',
                (peak, bins[-1] * 0.9),
                textcoords='data',
                color='cyan',
                rotation=90,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
            )

    return


######################################################################
# Startup and main routines
######################################################################


def shaper_calibration(
    lognames: List[str],
    klipperdir: str = '~/klipper',
    max_smoothing: Optional[float] = None,
    scv: float = 5.0,
    max_freq: float = 200.0,
    accel_per_hz: Optional[float] = None,
    st_version: str = 'unknown',
) -> plt.Figure:
    global shaper_calibrate
    shaper_calibrate = setup_klipper_import(klipperdir)

    # Parse data from the log files while ignoring CSV in the wrong format
    datas = [data for data in (parse_log(fn) for fn in lognames) if data is not None]
    if len(datas) > 1:
        ConsoleOutput.print('Warning: incorrect number of .csv files detected. Only the first one will be used!')

    # Compute shapers, PSD outputs and spectrogram
    klipper_shaper_choice, shapers, calibration_data, fr, zeta, compat = calibrate_shaper(
        datas[0], max_smoothing, scv, max_freq
    )
    pdata, bins, t = compute_spectrogram(datas[0])
    del datas

    # Select only the relevant part of the PSD data
    freqs = calibration_data.freq_bins
    calibration_data.psd_sum = calibration_data.psd_sum[freqs <= max_freq]
    calibration_data.psd_x = calibration_data.psd_x[freqs <= max_freq]
    calibration_data.psd_y = calibration_data.psd_y[freqs <= max_freq]
    calibration_data.psd_z = calibration_data.psd_z[freqs <= max_freq]
    calibration_data.freqs = freqs[freqs <= max_freq]

    # Peak detection algorithm
    peaks_threshold = [
        PEAKS_DETECTION_THRESHOLD * calibration_data.psd_sum.max(),
        PEAKS_EFFECT_THRESHOLD * calibration_data.psd_sum.max(),
    ]
    num_peaks, peaks, peaks_freqs = detect_peaks(calibration_data.psd_sum, calibration_data.freqs, peaks_threshold[0])

    # Print the peaks info in the console
    peak_freqs_formated = ['{:.1f}'.format(f) for f in peaks_freqs]
    num_peaks_above_effect_threshold = np.sum(calibration_data.psd_sum[peaks] > peaks_threshold[1])
    ConsoleOutput.print(
        '\nPeaks detected on the graph: %d @ %s Hz (%d above effect threshold)'
        % (num_peaks, ', '.join(map(str, peak_freqs_formated)), num_peaks_above_effect_threshold)
    )

    # Create graph layout
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        gridspec_kw={
            'height_ratios': [4, 3],
            'bottom': 0.050,
            'top': 0.890,
            'left': 0.085,
            'right': 0.966,
            'hspace': 0.169,
            'wspace': 0.200,
        },
    )
    fig.set_size_inches(8.3, 11.6)

    # Add a title with some test info
    title_line1 = 'INPUT SHAPER CALIBRATION TOOL'
    fig.text(
        0.12, 0.965, title_line1, ha='left', va='bottom', fontsize=20, color=KLIPPAIN_COLORS['purple'], weight='bold'
    )
    try:
        filename_parts = (lognames[0].split('/')[-1]).split('_')
        dt = datetime.strptime(f'{filename_parts[1]} {filename_parts[2]}', '%Y%m%d %H%M%S')
        title_line2 = dt.strftime('%x %X') + ' -- ' + filename_parts[3].upper().split('.')[0] + ' axis'
        if compat:
            title_line3 = '| Older Klipper version detected, damping ratio'
            title_line4 = '| and SCV are not used for filter recommendations!'
            title_line5 = f'| Accel per Hz used: {accel_per_hz} mm/s²/Hz' if accel_per_hz is not None else ''
        else:
            title_line3 = f'| Square corner velocity: {scv} mm/s'
            title_line4 = f'| Max allowed smoothing: {max_smoothing}'
            title_line5 = f'| Accel per Hz used: {accel_per_hz} mm/s²/Hz' if accel_per_hz is not None else ''
    except Exception:
        ConsoleOutput.print('Warning: CSV filename look to be different than expected (%s)' % (lognames[0]))
        title_line2 = lognames[0].split('/')[-1]
        title_line3 = ''
        title_line4 = ''
        title_line5 = ''
    fig.text(0.12, 0.957, title_line2, ha='left', va='top', fontsize=16, color=KLIPPAIN_COLORS['dark_purple'])
    fig.text(0.58, 0.963, title_line3, ha='left', va='top', fontsize=10, color=KLIPPAIN_COLORS['dark_purple'])
    fig.text(0.58, 0.948, title_line4, ha='left', va='top', fontsize=10, color=KLIPPAIN_COLORS['dark_purple'])
    fig.text(0.58, 0.933, title_line5, ha='left', va='top', fontsize=10, color=KLIPPAIN_COLORS['dark_purple'])

    # Plot the graphs
    plot_freq_response(
        ax1, calibration_data, shapers, klipper_shaper_choice, peaks, peaks_freqs, peaks_threshold, fr, zeta, max_freq
    )
    plot_spectrogram(ax2, t, bins, pdata, peaks_freqs, max_freq)

    # Adding a small Klippain logo to the top left corner of the figure
    ax_logo = fig.add_axes([0.001, 0.8995, 0.1, 0.1], anchor='NW')
    ax_logo.imshow(plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'klippain.png')))
    ax_logo.axis('off')

    # Adding Shake&Tune version in the top right corner
    if st_version != 'unknown':
        fig.text(0.995, 0.985, st_version, ha='right', va='bottom', fontsize=8, color=KLIPPAIN_COLORS['purple'])

    return fig


def main():
    # Parse command-line arguments
    usage = '%prog [options] <logs>'
    opts = optparse.OptionParser(usage)
    opts.add_option('-o', '--output', type='string', dest='output', default=None, help='filename of output graph')
    opts.add_option('-f', '--max_freq', type='float', default=200.0, help='maximum frequency to graph')
    opts.add_option('-s', '--max_smoothing', type='float', default=None, help='maximum shaper smoothing to allow')
    opts.add_option(
        '--scv', '--square_corner_velocity', type='float', dest='scv', default=5.0, help='square corner velocity'
    )
    opts.add_option('--accel_per_hz', type='float', default=None, help='accel_per_hz used during the measurement')
    opts.add_option(
        '-k', '--klipper_dir', type='string', dest='klipperdir', default='~/klipper', help='main klipper directory'
    )
    options, args = opts.parse_args()
    if len(args) < 1:
        opts.error('Incorrect number of arguments')
    if options.output is None:
        opts.error('You must specify an output file.png to use the script (option -o)')
    if options.max_smoothing is not None and options.max_smoothing < 0.05:
        opts.error('Too small max_smoothing specified (must be at least 0.05)')

    fig = shaper_calibration(
        args, options.klipperdir, options.max_smoothing, options.scv, options.max_freq, options.accel_per_hz, 'unknown'
    )
    fig.savefig(options.output, dpi=150)


if __name__ == '__main__':
    main()
