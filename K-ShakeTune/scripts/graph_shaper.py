#!/usr/bin/env python3

#################################################
######## INPUT SHAPER CALIBRATION SCRIPT ########
#################################################
# Derived from the calibrate_shaper.py official Klipper script
# Copyright (C) 2020  Dmitry Butyugin <dmbutyugin@google.com>
# Copyright (C) 2020  Kevin O'Connor <kevin@koconnor.net>
# Written by Frix_x#0161 #

# Be sure to make this script executable using SSH: type 'chmod +x ./graph_shaper.py' when in the folder!

#####################################################################
################ !!! DO NOT EDIT BELOW THIS LINE !!! ################
#####################################################################

import optparse, matplotlib, sys, importlib, os, math
import numpy as np
import matplotlib.pyplot, matplotlib.dates, matplotlib.font_manager
import matplotlib.ticker, matplotlib.gridspec
from datetime import datetime

matplotlib.use('Agg')

from locale_utils import set_locale, print_with_c_locale
from common_func import compute_spectrogram, detect_peaks


PEAKS_DETECTION_THRESHOLD = 0.05
PEAKS_EFFECT_THRESHOLD = 0.12
SPECTROGRAM_LOW_PERCENTILE_FILTER = 5
MAX_SMOOTHING = 0.1

KLIPPAIN_COLORS = {
    "purple": "#70088C",
    "dark_purple": "#150140",
    "dark_orange": "#F24130"
}


######################################################################
# Computation
######################################################################

# Find the best shaper parameters using Klipper's official algorithm selection
def calibrate_shaper_with_damping(datas, max_smoothing):
    helper = shaper_calibrate.ShaperCalibrate(printer=None)

    calibration_data = helper.process_accelerometer_data(datas[0])
    for data in datas[1:]:
        calibration_data.add_data(helper.process_accelerometer_data(data))

    calibration_data.normalize_to_frequencies()

    shaper, all_shapers = helper.find_best_shaper(calibration_data, max_smoothing, print_with_c_locale)

    freqs = calibration_data.freq_bins
    psd = calibration_data.psd_sum
    fr, zeta = compute_damping_ratio(psd, freqs)

    print_with_c_locale("Recommended shaper is %s @ %.1f Hz" % (shaper.name, shaper.freq))
    print_with_c_locale("Axis has a main resonant frequency at %.1fHz with an estimated damping ratio of %.3f" % (fr, zeta))

    return shaper.name, all_shapers, calibration_data, fr, zeta


# Compute damping ratio by using the half power bandwidth method with interpolated frequencies
def compute_damping_ratio(psd, freqs):
    max_power_index = np.argmax(psd)
    fr = freqs[max_power_index]
    max_power = psd[max_power_index]

    half_power = max_power / math.sqrt(2)
    idx_below = np.where(psd[:max_power_index] <= half_power)[0][-1]
    idx_above = np.where(psd[max_power_index:] <= half_power)[0][0] + max_power_index
    freq_below_half_power = freqs[idx_below] + (half_power - psd[idx_below]) * (freqs[idx_below + 1] - freqs[idx_below]) / (psd[idx_below + 1] - psd[idx_below])
    freq_above_half_power = freqs[idx_above - 1] + (half_power - psd[idx_above - 1]) * (freqs[idx_above] - freqs[idx_above - 1]) / (psd[idx_above] - psd[idx_above - 1])

    bandwidth = freq_above_half_power - freq_below_half_power
    zeta = bandwidth / (2 * fr)

    return fr, zeta


######################################################################
# Graphing
######################################################################

def plot_freq_response_with_damping(ax, calibration_data, shapers, performance_shaper, fr, zeta, max_freq):
    freqs = calibration_data.freq_bins
    psd = calibration_data.psd_sum[freqs <= max_freq]
    px = calibration_data.psd_x[freqs <= max_freq]
    py = calibration_data.psd_y[freqs <= max_freq]
    pz = calibration_data.psd_z[freqs <= max_freq]
    freqs = freqs[freqs <= max_freq]

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('x-small')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([0, max_freq])
    ax.set_ylabel('Power spectral density')
    ax.set_ylim([0, psd.max() + psd.max() * 0.05])

    ax.plot(freqs, psd, label='X+Y+Z', color='purple')
    ax.plot(freqs, px, label='X', color='red')
    ax.plot(freqs, py, label='Y', color='green')
    ax.plot(freqs, pz, label='Z', color='blue')

    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    ax2 = ax.twinx()
    ax2.yaxis.set_visible(False)
    
    lowvib_shaper_vibrs = float('inf')
    lowvib_shaper = None
    lowvib_shaper_freq = None
    lowvib_shaper_accel = 0
    
    # Draw the shappers curves and add their specific parameters in the legend
    # This adds also a way to find the best shaper with a low level of vibrations (with a resonable level of smoothing)
    for shaper in shapers:
        shaper_max_accel = round(shaper.max_accel / 100.) * 100.
        label = "%s (%.1f Hz, vibr=%.1f%%, sm~=%.2f, accel<=%.f)" % (
                shaper.name.upper(), shaper.freq,
                shaper.vibrs * 100., shaper.smoothing,
                shaper_max_accel)
        ax2.plot(freqs, shaper.vals, label=label, linestyle='dotted')

        # Get the performance shaper
        if shaper.name == performance_shaper:
            performance_shaper_freq = shaper.freq
            performance_shaper_vibr = shaper.vibrs * 100.
            performance_shaper_vals = shaper.vals

        # Get the low vibration shaper
        if (shaper.vibrs * 100 < lowvib_shaper_vibrs or (shaper.vibrs * 100 == lowvib_shaper_vibrs and shaper_max_accel > lowvib_shaper_accel)) and shaper.smoothing < MAX_SMOOTHING:
            lowvib_shaper_accel = shaper_max_accel
            lowvib_shaper = shaper.name
            lowvib_shaper_freq = shaper.freq
            lowvib_shaper_vibrs = shaper.vibrs * 100
            lowvib_shaper_vals = shaper.vals

    # User recommendations are added to the legend: one is Klipper's original suggestion that is usually good for performances
    # and the other one is the custom "low vibration" recommendation that looks for a suitable shaper that doesn't have excessive
    # smoothing and that have a lower vibration level. If both recommendation are the same shaper, or if no suitable "low
    # vibration" shaper is found, then only a single line as the "best shaper" recommendation is added to the legend
    if lowvib_shaper != None and lowvib_shaper != performance_shaper and lowvib_shaper_vibrs <= performance_shaper_vibr:
        ax2.plot([], [], ' ', label="Recommended performance shaper: %s @ %.1f Hz" % (performance_shaper.upper(), performance_shaper_freq))
        ax.plot(freqs, psd * performance_shaper_vals, label='With %s applied' % (performance_shaper.upper()), color='cyan')
        ax2.plot([], [], ' ', label="Recommended low vibrations shaper: %s @ %.1f Hz" % (lowvib_shaper.upper(), lowvib_shaper_freq))
        ax.plot(freqs, psd * lowvib_shaper_vals, label='With %s applied' % (lowvib_shaper.upper()), color='lime')
    else:
        ax2.plot([], [], ' ', label="Recommended best shaper: %s @ %.1f Hz" % (performance_shaper.upper(), performance_shaper_freq))
        ax.plot(freqs, psd * performance_shaper_vals, label='With %s applied' % (performance_shaper.upper()), color='cyan')

    # And the estimated damping ratio is finally added at the end of the legend
    ax2.plot([], [], ' ', label="Estimated damping ratio (ζ): %.3f" % (zeta))

    # Draw the detected peaks and name them
    # This also draw the detection threshold and warning threshold (aka "effect zone")
    peaks_warning_threshold = PEAKS_DETECTION_THRESHOLD * psd.max()
    peaks_effect_threshold = PEAKS_EFFECT_THRESHOLD * psd.max()
    num_peaks, peaks, peaks_freqs = detect_peaks(psd, freqs, peaks_warning_threshold)
    
    peak_freqs_formated = ["{:.1f}".format(f) for f in peaks_freqs]
    num_peaks_above_effect_threshold = np.sum(psd[peaks] > peaks_effect_threshold)
    print_with_c_locale("Peaks detected on the graph: %d @ %s Hz (%d above effect threshold)" % (num_peaks, ", ".join(map(str, peak_freqs_formated)), num_peaks_above_effect_threshold))

    ax.plot(peaks_freqs, psd[peaks], "x", color='black', markersize=8)
    for idx, peak in enumerate(peaks):
        if psd[peak] > peaks_effect_threshold:
            fontcolor = 'red'
            fontweight = 'bold'
        else:
            fontcolor = 'black'
            fontweight = 'normal'
        ax.annotate(f"{idx+1}", (freqs[peak], psd[peak]), 
                    textcoords="offset points", xytext=(8, 5), 
                    ha='left', fontsize=13, color=fontcolor, weight=fontweight)
    ax.axhline(y=peaks_warning_threshold, color='black', linestyle='--', linewidth=0.5)
    ax.axhline(y=peaks_effect_threshold, color='black', linestyle='--', linewidth=0.5)
    ax.fill_between(freqs, 0, peaks_warning_threshold, color='green', alpha=0.15, label='Relax Region')
    ax.fill_between(freqs, peaks_warning_threshold, peaks_effect_threshold, color='orange', alpha=0.2, label='Warning Region')


    # Add the main resonant frequency and damping ratio of the axis to the graph title
    ax.set_title("Axis Frequency Profile (ω0=%.1fHz, ζ=%.3f)" % (fr, zeta), fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.legend(loc='upper left', prop=fontP)
    ax2.legend(loc='upper right', prop=fontP)

    return peaks_freqs


# Plot a time-frequency spectrogram to see how the system respond over time during the
# resonnance test. This can highlight hidden spots from the standard PSD graph from other harmonics
def plot_spectrogram(ax, data, peaks, max_freq):
    pdata, bins, t = compute_spectrogram(data)

    # We need to normalize the data to get a proper signal on the spectrogram
    # However, while using "LogNorm" provide too much background noise, using
    # "Normalize" make only the resonnance appearing and hide interesting elements
    # So we need to filter out the lower part of the data (ie. find the proper vmin for LogNorm)
    vmin_value = np.percentile(pdata, SPECTROGRAM_LOW_PERCENTILE_FILTER)

    ax.set_title("Time-Frequency Spectrogram", fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.pcolormesh(t, bins, pdata.T, norm=matplotlib.colors.LogNorm(vmin=vmin_value),
                  cmap='inferno', shading='gouraud')

    # Add peaks lines in the spectrogram to get hint from peaks found in the first graph
    if peaks is not None:
        for idx, peak in enumerate(peaks):
            ax.axvline(peak, color='cyan', linestyle='dotted', linewidth=0.75)
            ax.annotate(f"Peak {idx+1}", (peak, t[-1]*0.9), 
                        textcoords="data", color='cyan', rotation=90, fontsize=10,
                        verticalalignment='top', horizontalalignment='right')

    ax.set_xlim([0., max_freq])
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Frequency (Hz)')

    return


######################################################################
# Startup and main routines
######################################################################

def parse_log(logname):
    with open(logname) as f:
        for header in f:
            if not header.startswith('#'):
                break
        if not header.startswith('freq,psd_x,psd_y,psd_z,psd_xyz'):
            # Raw accelerometer data
            return np.loadtxt(logname, comments='#', delimiter=',')
    # Power spectral density data or shaper calibration data
    raise ValueError("File %s does not contain raw accelerometer data and therefore "
               "is not supported by this script. Please use the official Klipper "
               "calibrate_shaper.py script to process it instead." % (logname,))


def setup_klipper_import(kdir):
    global shaper_calibrate
    kdir = os.path.expanduser(kdir)
    sys.path.append(os.path.join(kdir, 'klippy'))
    shaper_calibrate = importlib.import_module('.shaper_calibrate', 'extras')


def shaper_calibration(lognames, klipperdir="~/klipper", max_smoothing=None, max_freq=200.):
    set_locale()
    setup_klipper_import(klipperdir)

    # Parse data
    datas = [parse_log(fn) for fn in lognames]

    # Calibrate shaper and generate outputs
    performance_shaper, shapers, calibration_data, fr, zeta = calibrate_shaper_with_damping(datas, max_smoothing)

    fig = matplotlib.pyplot.figure()
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[4, 3])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Add title
    title_line1 = "INPUT SHAPER CALIBRATION TOOL"
    fig.text(0.12, 0.965, title_line1, ha='left', va='bottom', fontsize=20, color=KLIPPAIN_COLORS['purple'], weight='bold')
    try:
        filename_parts = (lognames[0].split('/')[-1]).split('_')
        dt = datetime.strptime(f"{filename_parts[1]} {filename_parts[2]}", "%Y%m%d %H%M%S")
        title_line2 = dt.strftime('%x %X') + ' -- ' + filename_parts[3].upper().split('.')[0] + ' axis'
    except:
        print_with_c_locale("Warning: CSV filename look to be different than expected (%s)" % (lognames[0]))
        title_line2 = lognames[0].split('/')[-1]
    fig.text(0.12, 0.957, title_line2, ha='left', va='top', fontsize=16, color=KLIPPAIN_COLORS['dark_purple'])

    # Plot the graphs
    peaks = plot_freq_response_with_damping(ax1, calibration_data, shapers, performance_shaper, fr, zeta, max_freq)
    plot_spectrogram(ax2, datas[0], peaks, max_freq)

    fig.set_size_inches(8.3, 11.6)
    fig.tight_layout()
    fig.subplots_adjust(top=0.89)

    # Adding a small Klippain logo to the top left corner of the figure
    ax_logo = fig.add_axes([0.001, 0.899, 0.1, 0.1], anchor='NW', zorder=-1)
    ax_logo.imshow(matplotlib.pyplot.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'klippain.png')))
    ax_logo.axis('off')

    return fig


def main():
    # Parse command-line arguments
    usage = "%prog [options] <logs>"
    opts = optparse.OptionParser(usage)
    opts.add_option("-o", "--output", type="string", dest="output",
                    default=None, help="filename of output graph")
    opts.add_option("-f", "--max_freq", type="float", default=200.,
                    help="maximum frequency to graph")
    opts.add_option("-s", "--max_smoothing", type="float", default=None,
                    help="maximum shaper smoothing to allow")
    opts.add_option("-k", "--klipper_dir", type="string", dest="klipperdir",
                    default="~/klipper", help="main klipper directory")
    options, args = opts.parse_args()
    if len(args) < 1:
        opts.error("Incorrect number of arguments")
    if options.output is None:
        opts.error("You must specify an output file.png to use the script (option -o)")
    if options.max_smoothing is not None and options.max_smoothing < 0.05:
        opts.error("Too small max_smoothing specified (must be at least 0.05)")

    fig = shaper_calibration(args, options.klipperdir, options.max_smoothing, options.max_freq)
    fig.savefig(options.output)


if __name__ == '__main__':
    main()
