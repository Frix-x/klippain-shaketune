# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: static_graph_creator.py
# Description: Implements a static frequency profile measurement script for Shake&Tune to diagnose mechanical
#              issues, including computation and graphing functions for 3D printer vibration analysis.


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

from ..helpers.common_func import compute_spectrogram, parse_log
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


class StaticGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config, 'static frequency')
        self._freq: Optional[float] = None
        self._duration: Optional[float] = None
        self._accel_per_hz: Optional[float] = None

    def configure(self, freq: float, duration: float, accel_per_hz: Optional[float] = None) -> None:
        self._freq = freq
        self._duration = duration
        self._accel_per_hz = accel_per_hz

    def create_graph(self) -> None:
        if not self._freq or not self._duration or not self._accel_per_hz:
            raise ValueError('freq, duration and accel_per_hz must be set to create the static frequency graph!')

        lognames = self._move_and_prepare_files(
            glob_pattern='shaketune-staticfreq_*.csv',
            min_files_required=1,
            custom_name_func=lambda f: f.stem.split('_')[1].upper(),
        )
        fig = static_frequency_tool(
            lognames=[str(path) for path in lognames],
            klipperdir=str(self._config.klipper_folder),
            freq=self._freq,
            duration=self._duration,
            max_freq=200.0,
            accel_per_hz=self._accel_per_hz,
            st_version=self._version,
        )
        self._save_figure_and_cleanup(fig, lognames, lognames[0].stem.split('_')[-1])

    def clean_old_files(self, keep_results: int = 3) -> None:
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)
        if len(files) <= keep_results:
            return  # No need to delete any files
        for old_file in files[keep_results:]:
            csv_file = old_file.with_suffix('.csv')
            csv_file.unlink(missing_ok=True)
            old_file.unlink()


######################################################################
# Graphing
######################################################################


def plot_spectrogram(ax: plt.Axes, t: np.ndarray, bins: np.ndarray, pdata: np.ndarray, max_freq: float) -> None:
    ax.set_title('Time-Frequency Spectrogram', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')

    vmin_value = np.percentile(pdata, SPECTROGRAM_LOW_PERCENTILE_FILTER)

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

    return


def plot_energy_accumulation(ax: plt.Axes, t: np.ndarray, bins: np.ndarray, pdata: np.ndarray) -> None:
    # Integrate the energy over the frequency bins for each time step and plot this vertically
    ax.plot(np.trapz(pdata, t, axis=0), bins, color=KLIPPAIN_COLORS['orange'])
    ax.set_title('Vibrations', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_xlabel('Cumulative Energy')
    ax.set_ylabel('Time (s)')
    ax.set_ylim([bins[0], bins[-1]])

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')
    # ax.legend()


######################################################################
# Startup and main routines
######################################################################


def static_frequency_tool(
    lognames: List[str],
    klipperdir: str = '~/klipper',
    freq: Optional[float] = None,
    duration: Optional[float] = None,
    max_freq: float = 500.0,
    accel_per_hz: Optional[float] = None,
    st_version: str = 'unknown',
) -> plt.Figure:
    if freq is None or duration is None:
        raise ValueError('Error: missing frequency or duration parameters!')

    datas = [data for data in (parse_log(fn) for fn in lognames) if data is not None]
    if len(datas) == 0:
        raise ValueError('No valid data found in the provided CSV files!')
    if len(datas) > 1:
        ConsoleOutput.print('Warning: incorrect number of .csv files detected. Only the first one will be used!')

    pdata, bins, t = compute_spectrogram(datas[0])
    del datas

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

    title_line1 = 'STATIC FREQUENCY HELPER TOOL'
    fig.text(
        0.060, 0.947, title_line1, ha='left', va='bottom', fontsize=20, color=KLIPPAIN_COLORS['purple'], weight='bold'
    )
    try:
        filename_parts = (lognames[0].split('/')[-1]).split('_')
        dt = datetime.strptime(f'{filename_parts[1]} {filename_parts[2]}', '%Y%m%d %H%M%S')
        title_line2 = dt.strftime('%x %X') + ' -- ' + filename_parts[3].upper().split('.')[0] + ' axis'
        title_line3 = f'| Maintained frequency: {freq}Hz for {duration}s'
        title_line4 = f'| Accel per Hz used: {accel_per_hz} mm/s²/Hz' if accel_per_hz is not None else ''
    except Exception:
        ConsoleOutput.print(f'Warning: CSV filename look to be different than expected ({lognames[0]})')
        title_line2 = lognames[0].split('/')[-1]
        title_line3 = ''
        title_line4 = ''
    fig.text(0.060, 0.939, title_line2, ha='left', va='top', fontsize=16, color=KLIPPAIN_COLORS['dark_purple'])
    fig.text(0.55, 0.985, title_line3, ha='left', va='top', fontsize=14, color=KLIPPAIN_COLORS['dark_purple'])
    fig.text(0.55, 0.950, title_line4, ha='left', va='top', fontsize=11, color=KLIPPAIN_COLORS['dark_purple'])

    plot_spectrogram(ax1, t, bins, pdata, max_freq)
    plot_energy_accumulation(ax3, t, bins, pdata)

    ax_logo = fig.add_axes([0.001, 0.894, 0.105, 0.105], anchor='NW')
    ax_logo.imshow(plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'klippain.png')))
    ax_logo.axis('off')

    if st_version != 'unknown':
        fig.text(0.995, 0.980, st_version, ha='right', va='bottom', fontsize=8, color=KLIPPAIN_COLORS['purple'])

    return fig


def main():
    usage = '%prog [options] <logs>'
    opts = optparse.OptionParser(usage)
    opts.add_option('-o', '--output', type='string', dest='output', default=None, help='filename of output graph')
    opts.add_option('-f', '--freq', type='float', default=None, help='frequency maintained during the measurement')
    opts.add_option('-d', '--duration', type='float', default=None, help='duration of the measurement')
    opts.add_option('--max_freq', type='float', default=500.0, help='maximum frequency to graph')
    opts.add_option('--accel_per_hz', type='float', default=None, help='accel_per_hz used during the measurement')
    opts.add_option(
        '-k', '--klipper_dir', type='string', dest='klipperdir', default='~/klipper', help='main klipper directory'
    )
    options, args = opts.parse_args()
    if len(args) < 1:
        opts.error('Incorrect number of arguments')
    if options.output is None:
        opts.error('You must specify an output file.png to use the script (option -o)')

    fig = static_frequency_tool(
        args, options.klipperdir, options.freq, options.duration, options.max_freq, options.accel_per_hz, 'unknown'
    )
    fig.savefig(options.output, dpi=150)


if __name__ == '__main__':
    main()
