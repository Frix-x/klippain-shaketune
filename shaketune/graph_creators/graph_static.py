#!/usr/bin/env python3

import optparse
import os
from datetime import datetime

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

matplotlib.use('Agg')

from ..helpers.common_func import (
    compute_spectrogram,
    parse_log,
)
from ..helpers.console_output import ConsoleOutput

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


######################################################################
# Graphing
######################################################################


def plot_spectrogram(ax, t, bins, pdata, max_freq):
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


def plot_energy_accumulation(ax, t, bins, pdata):
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
    lognames,
    klipperdir='~/klipper',
    freq=None,
    duration=None,
    max_freq=500.0,
    accel_per_hz=None,
    st_version='unknown',
):
    if freq is None or duration is None:
        raise ValueError('Error: missing frequency or duration parameters!')

    datas = [data for data in (parse_log(fn) for fn in lognames) if data is not None]
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
        ConsoleOutput.print('Warning: CSV filename look to be different than expected (%s)' % (lognames[0]))
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
