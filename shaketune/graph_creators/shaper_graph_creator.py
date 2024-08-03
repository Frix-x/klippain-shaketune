# Shake&Tune: 3D printer analysis tools
#
# Derived from the calibrate_shaper.py official Klipper script
# Copyright (C) 2020  Dmitry Butyugin <dmbutyugin@google.com>
# Copyright (C) 2020  Kevin O'Connor <kevin@koconnor.net>
# Copyright (C) 2022 - 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: shaper_graph_creator.py
# Description: Implements the input shaper calibration script for Shake&Tune,
#              including computation and graphing functions for 3D printer vibration analysis.


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
from typing import Dict, List, Optional

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.interpolate import interp1d

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
MAX_VIBRATIONS_PLOTTED = 80.0
MAX_VIBRATIONS_PLOTTED_ZOOM = 1.25  # 1.25x max vibs values from the standard filters selection
SMOOTHING_TESTS = 10  # Number of smoothing values to test (it will significantly increase the computation time)
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

    # We compute the damping ratio using the Klipper's default value if it fails
    fr, zeta, _, _ = compute_mechanical_parameters(calibration_data.psd_sum, calibration_data.freq_bins)
    zeta = zeta if zeta is not None else 0.1

    compat = False
    try:
        k_shaper_choice, all_shapers = helper.find_best_shaper(
            calibration_data,
            shapers=None,
            damping_ratio=zeta,
            scv=scv,
            shaper_freqs=None,
            max_smoothing=max_smoothing,
            test_damping_ratios=None,
            max_freq=max_freq,
            logger=None,
        )
        ConsoleOutput.print(
            (
                f'Detected a square corner velocity of {scv:.1f} and a damping ratio of {zeta:.3f}. '
                'These values will be used to compute the input shaper filter recommendations'
            )
        )
    except TypeError:
        ConsoleOutput.print(
            (
                '[WARNING] You seem to be using an older version of Klipper that is not compatible with all the latest '
                'Shake&Tune features!\nShake&Tune now runs in compatibility mode: be aware that the results may be '
                'slightly off, since the real damping ratio cannot be used to craft accurate filter recommendations'
            )
        )
        compat = True
        k_shaper_choice, all_shapers = helper.find_best_shaper(calibration_data, max_smoothing, None)

    # If max_smoothing is not None, we run the same computation but without a smoothing value
    # to get the max smoothing values from the filters and create the testing list
    all_shapers_nosmoothing = None
    if max_smoothing is not None:
        if compat:
            _, all_shapers_nosmoothing = helper.find_best_shaper(calibration_data, None, None)
        else:
            _, all_shapers_nosmoothing = helper.find_best_shaper(
                calibration_data,
                shapers=None,
                damping_ratio=zeta,
                scv=scv,
                shaper_freqs=None,
                max_smoothing=None,
                test_damping_ratios=None,
                max_freq=max_freq,
                logger=None,
            )

    # Then we iterate over the all_shaperts_nosmoothing list to get the max of the smoothing values
    max_smoothing = 0.0
    if all_shapers_nosmoothing is not None:
        for shaper in all_shapers_nosmoothing:
            if shaper.smoothing > max_smoothing:
                max_smoothing = shaper.smoothing
    else:
        for shaper in all_shapers:
            if shaper.smoothing > max_smoothing:
                max_smoothing = shaper.smoothing

    # Then we create a list of smoothing values to test (no need to test the max smoothing value as it was already tested)
    smoothing_test_list = np.linspace(0.001, max_smoothing, SMOOTHING_TESTS)[:-1]
    additional_all_shapers = {}
    for smoothing in smoothing_test_list:
        if compat:
            _, all_shapers_bis = helper.find_best_shaper(calibration_data, smoothing, None)
        else:
            _, all_shapers_bis = helper.find_best_shaper(
                calibration_data,
                shapers=None,
                damping_ratio=zeta,
                scv=scv,
                shaper_freqs=None,
                max_smoothing=smoothing,
                test_damping_ratios=None,
                max_freq=max_freq,
                logger=None,
            )
        additional_all_shapers[f'sm_{smoothing}'] = all_shapers_bis
    additional_all_shapers['max_smoothing'] = (
        all_shapers_nosmoothing if all_shapers_nosmoothing is not None else all_shapers
    )

    return k_shaper_choice.name, all_shapers, additional_all_shapers, calibration_data, fr, zeta, max_smoothing, compat


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
) -> Dict[str, List[Dict[str, str]]]:
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

    shaper_table_data = {
        'shapers': [],
        'recommendations': [],
        'damping_ratio': zeta,
    }

    # Draw the shappers curves and add their specific parameters in the legend
    perf_shaper_choice = None
    perf_shaper_vals = None
    perf_shaper_freq = None
    perf_shaper_accel = 0
    for shaper in shapers:
        ax2.plot(freqs, shaper.vals, label=shaper.name.upper(), linestyle='dotted')

        shaper_info = {
            'type': shaper.name.upper(),
            'frequency': shaper.freq,
            'vibrations': shaper.vibrs,
            'smoothing': shaper.smoothing,
            'max_accel': shaper.max_accel,
        }
        shaper_table_data['shapers'].append(shaper_info)

        # Get the Klipper recommended shaper (usually it's a good low vibration compromise)
        if shaper.name == klipper_shaper_choice:
            klipper_shaper_freq = shaper.freq
            klipper_shaper_vals = shaper.vals
            klipper_shaper_accel = shaper.max_accel

        # Find the shaper with the highest accel but with vibrs under MAX_VIBRATIONS as it's
        # a good performance compromise when injecting the SCV and damping ratio in the computation
        if perf_shaper_accel < shaper.max_accel and shaper.vibrs * 100 < MAX_VIBRATIONS:
            perf_shaper_choice = shaper.name
            perf_shaper_accel = shaper.max_accel
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
        perf_shaper_string = f'Recommended for performance: {perf_shaper_choice.upper()} @ {perf_shaper_freq:.1f} Hz'
        lowvibr_shaper_string = (
            f'Recommended for low vibrations: {klipper_shaper_choice.upper()} @ {klipper_shaper_freq:.1f} Hz'
        )
        shaper_table_data['recommendations'].append(perf_shaper_string)
        shaper_table_data['recommendations'].append(lowvibr_shaper_string)
        ConsoleOutput.print(f'{perf_shaper_string} (with a damping ratio of {zeta:.3f})')
        ConsoleOutput.print(f'{lowvibr_shaper_string} (with a damping ratio of {zeta:.3f})')
        ax.plot(
            freqs,
            psd * perf_shaper_vals,
            label=f'With {perf_shaper_choice.upper()} applied',
            color='cyan',
        )
        ax.plot(
            freqs,
            psd * klipper_shaper_vals,
            label=f'With {klipper_shaper_choice.upper()} applied',
            color='lime',
        )
    else:
        shaper_string = f'Recommended best shaper: {klipper_shaper_choice.upper()} @ {klipper_shaper_freq:.1f} Hz'
        shaper_table_data['recommendations'].append(shaper_string)
        ConsoleOutput.print(f'{shaper_string} (with a damping ratio of {zeta:.3f})')
        ax.plot(
            freqs,
            psd * klipper_shaper_vals,
            label=f'With {klipper_shaper_choice.upper()} applied',
            color='cyan',
        )

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
        f'Axis Frequency Profile (ω0={fr:.1f}Hz, ζ={zeta:.3f})',
        fontsize=14,
        color=KLIPPAIN_COLORS['dark_orange'],
        weight='bold',
    )
    ax.legend(loc='upper left', prop=fontP)
    ax2.legend(loc='upper right', prop=fontP)

    return shaper_table_data


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


def plot_smoothing_vs_accel(
    ax: plt.Axes,
    shaper_table_data: Dict[str, List[Dict[str, str]]],
    additional_shapers: Dict[str, List[Dict[str, str]]],
) -> None:
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('x-small')

    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

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
            max_accel_limit, max(d['max_accel'] for d in data if d['vibrs'] <= MAX_VIBRATIONS_PLOTTED)
        )
        max_accel_limit_zoom = max(
            max_accel_limit_zoom,
            max(d['max_accel'] for d in data if d['vibrs'] <= max_shaper_vibrations * MAX_VIBRATIONS_PLOTTED_ZOOM),
        )

    # Add a zoom axes on the graph to show details at low vibrations
    zoomed_window = np.clip(max_shaper_vibrations * MAX_VIBRATIONS_PLOTTED_ZOOM, 0, 20)
    axins = ax.inset_axes(
        [0.575, 0.125, 0.40, 0.45],
        xlim=(min_accel_limit * 0.95, max_accel_limit_zoom * 1.1),
        ylim=(-0.5, zoomed_window),
    )
    ax.indicate_inset_zoom(axins, edgecolor=KLIPPAIN_COLORS['purple'], linewidth=3)
    axins.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
    axins.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axins.grid(which='major', color='grey')
    axins.grid(which='minor', color='lightgrey')

    # Draw the green zone on both axes to highlight the low vibrations zone
    number_of_interpolated_points = 100
    x_fill = np.linspace(min_accel_limit * 0.95, max_accel_limit * 1.1, number_of_interpolated_points)
    y_fill = np.full_like(x_fill, 5.0)
    ax.axhline(y=5.0, color='black', linestyle='--', linewidth=0.5)
    ax.fill_between(x_fill, -0.5, y_fill, color='green', alpha=0.15)
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

        ax.plot(max_accel_fine, vibrs_fine, label=f'{shaper_type}', zorder=10)
        axins.plot(max_accel_fine, vibrs_fine, label=f'{shaper_type}', zorder=15)
        max_vibrations = max(max_vibrations, max(vibrs_fine))

    ax.set_xlabel('Max Acceleration')
    ax.set_ylabel('Remaining Vibrations (%)')
    ax.set_xlim([min_accel_limit * 0.95, max_accel_limit * 1.1])
    ax.set_ylim([-0.5, np.clip(max_vibrations * 1.05, 50, MAX_VIBRATIONS_PLOTTED)])
    ax.set_title(
        'Filters performances over acceleration',
        fontsize=14,
        color=KLIPPAIN_COLORS['dark_orange'],
        weight='bold',
    )
    ax.legend(loc='best', prop=fontP)


def print_shaper_table(fig: plt.Figure, shaper_table_data: Dict[str, List[Dict[str, str]]]) -> None:
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
        color=KLIPPAIN_COLORS['purple'],
    )
    if len(shaper_table_data['recommendations']) == 1:
        fig.text(
            0.585,
            0.200,
            shaper_table_data['recommendations'][0],
            fontsize=14,
            color=KLIPPAIN_COLORS['red_pink'],
        )
    elif len(shaper_table_data['recommendations']) == 2:
        fig.text(
            0.585,
            0.200,
            shaper_table_data['recommendations'][0],
            fontsize=14,
            color=KLIPPAIN_COLORS['red_pink'],
        )
        fig.text(
            0.585,
            0.175,
            shaper_table_data['recommendations'][1],
            fontsize=14,
            color=KLIPPAIN_COLORS['red_pink'],
        )


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
    if len(datas) == 0:
        raise ValueError('No valid data found in the provided CSV files!')
    if len(datas) > 1:
        ConsoleOutput.print('Warning: incorrect number of .csv files detected. Only the first one will be used!')

    # Compute shapers, PSD outputs and spectrogram
    klipper_shaper_choice, shapers, additional_shapers, calibration_data, fr, zeta, max_smoothing_computed, compat = (
        calibrate_shaper(datas[0], max_smoothing, scv, max_freq)
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
        f"Peaks detected on the graph: {num_peaks} @ {', '.join(map(str, peak_freqs_formated))} Hz ({num_peaks_above_effect_threshold} above effect threshold)"
    )

    # Create graph layout
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(
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
    ax4.remove()
    fig.set_size_inches(15, 11.6)

    # Add a title with some test info
    title_line1 = 'INPUT SHAPER CALIBRATION TOOL'
    fig.text(
        0.065, 0.965, title_line1, ha='left', va='bottom', fontsize=20, color=KLIPPAIN_COLORS['purple'], weight='bold'
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
            max_smoothing_string = (
                f'maximum ({max_smoothing_computed:0.3f})' if max_smoothing is None else f'{max_smoothing:0.3f}'
            )
            title_line3 = f'| Square corner velocity: {scv} mm/s'
            title_line4 = f'| Allowed smoothing: {max_smoothing_string}'
            title_line5 = f'| Accel per Hz used: {accel_per_hz} mm/s²/Hz' if accel_per_hz is not None else ''
    except Exception:
        ConsoleOutput.print(f'Warning: CSV filename look to be different than expected ({lognames[0]})')
        title_line2 = lognames[0].split('/')[-1]
        title_line3 = ''
        title_line4 = ''
        title_line5 = ''
    fig.text(0.065, 0.957, title_line2, ha='left', va='top', fontsize=16, color=KLIPPAIN_COLORS['dark_purple'])
    fig.text(0.50, 0.990, title_line3, ha='left', va='top', fontsize=14, color=KLIPPAIN_COLORS['dark_purple'])
    fig.text(0.50, 0.968, title_line4, ha='left', va='top', fontsize=14, color=KLIPPAIN_COLORS['dark_purple'])
    fig.text(0.501, 0.945, title_line5, ha='left', va='top', fontsize=10, color=KLIPPAIN_COLORS['dark_purple'])

    # Plot the graphs
    shaper_table_data = plot_freq_response(
        ax1, calibration_data, shapers, klipper_shaper_choice, peaks, peaks_freqs, peaks_threshold, fr, zeta, max_freq
    )
    plot_spectrogram(ax2, t, bins, pdata, peaks_freqs, max_freq)
    plot_smoothing_vs_accel(ax3, shaper_table_data, additional_shapers)

    print_shaper_table(fig, shaper_table_data)

    # Adding a small Klippain logo to the top left corner of the figure
    ax_logo = fig.add_axes([0.001, 0.924, 0.075, 0.075], anchor='NW')
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
