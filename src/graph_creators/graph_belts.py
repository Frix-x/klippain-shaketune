#!/usr/bin/env python3

#################################################
######## CoreXY BELTS CALIBRATION SCRIPT ########
#################################################
# Written by Frix_x#0161 #

import optparse
import os
from collections import namedtuple
from datetime import datetime

import matplotlib
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.interpolate import griddata

matplotlib.use('Agg')

from ..helpers.common_func import (
    compute_curve_similarity_factor,
    compute_spectrogram,
    detect_peaks,
    parse_log,
    setup_klipper_import,
)
from ..helpers.locale_utils import print_with_c_locale, set_locale

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # For paired peaks names

PEAKS_DETECTION_THRESHOLD = 0.20
CURVE_SIMILARITY_SIGMOID_K = 0.6
DC_GRAIN_OF_SALT_FACTOR = 0.75
DC_THRESHOLD_METRIC = 1.5e9
DC_MAX_UNPAIRED_PEAKS_ALLOWED = 4

# Define the SignalData namedtuple
SignalData = namedtuple('CalibrationData', ['freqs', 'psd', 'peaks', 'paired_peaks', 'unpaired_peaks'])

KLIPPAIN_COLORS = {
    'purple': '#70088C',
    'orange': '#FF8D32',
    'dark_purple': '#150140',
    'dark_orange': '#F24130',
    'red_pink': '#F2055C',
}


######################################################################
# Computation of the PSD graph
######################################################################


# This function create pairs of peaks that are close in frequency on two curves (that are known
# to be resonances points and must be similar on both belts on a CoreXY kinematic)
def pair_peaks(peaks1, freqs1, psd1, peaks2, freqs2, psd2):
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

    return paired_peaks, unpaired_peaks1, unpaired_peaks2


######################################################################
# Computation of the differential spectrogram
######################################################################


# Interpolate source_data (2D) to match target_x and target_y in order to
# get similar time and frequency dimensions for the differential spectrogram
def interpolate_2d(target_x, target_y, source_x, source_y, source_data):
    # Create a grid of points in the source and target space
    source_points = np.array([(x, y) for y in source_y for x in source_x])
    target_points = np.array([(x, y) for y in target_y for x in target_x])

    # Flatten the source data to match the flattened source points
    source_values = source_data.flatten()

    # Interpolate and reshape the interpolated data to match the target grid shape and replace NaN with zeros
    interpolated_data = griddata(source_points, source_values, target_points, method='nearest')
    interpolated_data = interpolated_data.reshape((len(target_y), len(target_x)))
    interpolated_data = np.nan_to_num(interpolated_data)

    return interpolated_data


# Main logic function to combine two similar spectrogram - ie. from both belts paths - by substracting signals in order to create
# a new composite spectrogram. This result of a divergent but mostly centered new spectrogram (center will be white) with some colored zones
# highlighting differences in the belts paths. The summative spectrogram is used for the MHI calculation.
def compute_combined_spectrogram(data1, data2):
    pdata1, bins1, t1 = compute_spectrogram(data1)
    pdata2, bins2, t2 = compute_spectrogram(data2)

    # Interpolate the spectrograms
    pdata2_interpolated = interpolate_2d(bins1, t1, bins2, t2, pdata2)

    # Combine them in two form: a summed diff for the MHI computation and a diverging diff for the spectrogram colors
    combined_sum = np.abs(pdata1 - pdata2_interpolated)
    combined_divergent = pdata1 - pdata2_interpolated

    return combined_sum, combined_divergent, bins1, t1


# Compute a composite and highly subjective value indicating the "mechanical health of the printer (0 to 100%)" that represent the
# likelihood of mechanical issues on the printer. It is based on the differential spectrogram sum of gradient, salted with a bit
# of the estimated similarity cross-correlation from compute_curve_similarity_factor() and with a bit of the number of unpaired peaks.
# This result in a percentage value quantifying the machine behavior around the main resonances that give an hint if only touching belt tension
# will give good graphs or if there is a chance of mechanical issues in the background (above 50% should be considered as probably problematic)
def compute_mhi(combined_data, similarity_coefficient, num_unpaired_peaks):
    # filtered_data = combined_data[combined_data > 100]
    filtered_data = np.abs(combined_data)

    # First compute a "total variability metric" based on the sum of the gradient that sum the magnitude of will emphasize regions of the
    # spectrogram where there are rapid changes in magnitude (like the edges of resonance peaks).
    total_variability_metric = np.sum(np.abs(np.gradient(filtered_data)))
    # Scale the metric to a percentage using the threshold (found empirically on a large number of user data shared to me)
    base_percentage = (np.log1p(total_variability_metric) / np.log1p(DC_THRESHOLD_METRIC)) * 100

    # Adjust the percentage based on the similarity_coefficient to add a grain of salt
    adjusted_percentage = base_percentage * (1 - DC_GRAIN_OF_SALT_FACTOR * (similarity_coefficient / 100))

    # Adjust the percentage again based on the number of unpaired peaks to add a second grain of salt
    peak_confidence = num_unpaired_peaks / DC_MAX_UNPAIRED_PEAKS_ALLOWED
    final_percentage = (1 - peak_confidence) * adjusted_percentage + peak_confidence * 100

    # Ensure the result lies between 0 and 100 by clipping the computed value
    final_percentage = np.clip(final_percentage, 0, 100)

    return final_percentage, mhi_lut(final_percentage)


# LUT to transform the MHI into a textual value easy to understand for the users of the script
def mhi_lut(mhi):
    ranges = [
        (0, 30, 'Excellent mechanical health'),
        (30, 45, 'Good mechanical health'),
        (45, 55, 'Acceptable mechanical health'),
        (55, 70, 'Potential signs of a mechanical issue'),
        (70, 85, 'Likely a mechanical issue'),
        (85, 100, 'Mechanical issue detected'),
    ]
    for lower, upper, message in ranges:
        if lower < mhi <= upper:
            return message

    return 'Error computing MHI value'


######################################################################
# Graphing
######################################################################


def plot_compare_frequency(ax, lognames, signal1, signal2, similarity_factor, max_freq):
    # Get the belt name for the legend to avoid putting the full file name
    signal1_belt = (lognames[0].split('/')[-1]).split('_')[-1][0]
    signal2_belt = (lognames[1].split('/')[-1]).split('_')[-1][0]

    if signal1_belt == 'A' and signal2_belt == 'B':
        signal1_belt += ' (axis 1,-1)'
        signal2_belt += ' (axis 1, 1)'
    elif signal1_belt == 'B' and signal2_belt == 'A':
        signal1_belt += ' (axis 1, 1)'
        signal2_belt += ' (axis 1,-1)'
    else:
        print_with_c_locale(
            "Warning: belts doesn't seem to have the correct name A and B (extracted from the filename.csv)"
        )

    # Plot the two belts PSD signals
    ax.plot(signal1.freqs, signal1.psd, label='Belt ' + signal1_belt, color=KLIPPAIN_COLORS['purple'])
    ax.plot(signal2.freqs, signal2.psd, label='Belt ' + signal2_belt, color=KLIPPAIN_COLORS['orange'])

    # Trace the "relax region" (also used as a threshold to filter and detect the peaks)
    psd_lowest_max = min(signal1.psd.max(), signal2.psd.max())
    peaks_warning_threshold = PEAKS_DETECTION_THRESHOLD * psd_lowest_max
    ax.axhline(y=peaks_warning_threshold, color='black', linestyle='--', linewidth=0.5)
    ax.fill_between(signal1.freqs, 0, peaks_warning_threshold, color='green', alpha=0.15, label='Relax Region')

    # Trace and annotate the peaks on the graph
    paired_peak_count = 0
    unpaired_peak_count = 0
    offsets_table_data = []

    for _, (peak1, peak2) in enumerate(signal1.paired_peaks):
        label = ALPHABET[paired_peak_count]
        amplitude_offset = abs(
            ((signal2.psd[peak2[0]] - signal1.psd[peak1[0]]) / max(signal1.psd[peak1[0]], signal2.psd[peak2[0]])) * 100
        )
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
    ax2.plot([], [], ' ', label=f'Estimated similarity: {similarity_factor:.1f}%')
    ax2.plot([], [], ' ', label=f'Number of unpaired peaks: {unpaired_peak_count}')

    # Setting axis parameters, grid and graph title
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([0, max_freq])
    ax.set_ylabel('Power spectral density')
    psd_highest_max = max(signal1.psd.max(), signal2.psd.max())
    ax.set_ylim([0, psd_highest_max + psd_highest_max * 0.05])

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax.set_title(
        'Belts Frequency Profiles (estimated similarity: {:.1f}%)'.format(similarity_factor),
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
            bbox=[0.66, 0.75, 0.33, 0.15],
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


def plot_difference_spectrogram(ax, signal1, signal2, t, bins, combined_divergent, textual_mhi, max_freq):
    ax.set_title('Differential Spectrogram', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.plot([], [], ' ', label=f'{textual_mhi} (experimental)')

    # Draw the differential spectrogram with a specific custom norm to get orange or purple values where there is signal or white near zeros
    # imgshow is better suited here than pcolormesh since its result is already rasterized and we doesn't need to keep vector graphics
    # when saving to a final .png file. Using it also allow to save ~150-200MB of RAM during the "fig.savefig" operation.
    colors = [
        KLIPPAIN_COLORS['dark_orange'],
        KLIPPAIN_COLORS['orange'],
        'white',
        KLIPPAIN_COLORS['purple'],
        KLIPPAIN_COLORS['dark_purple'],
    ]
    cm = matplotlib.colors.LinearSegmentedColormap.from_list(
        'klippain_divergent', list(zip([0, 0.25, 0.5, 0.75, 1], colors))
    )
    norm = matplotlib.colors.TwoSlopeNorm(vmin=np.min(combined_divergent), vcenter=0, vmax=np.max(combined_divergent))
    ax.imshow(
        combined_divergent.T,
        cmap=cm,
        norm=norm,
        aspect='auto',
        extent=[t[0], t[-1], bins[0], bins[-1]],
        interpolation='bilinear',
        origin='lower',
    )

    ax.set_xlabel('Frequency (hz)')
    ax.set_xlim([0.0, max_freq])
    ax.set_ylabel('Time (s)')
    ax.set_ylim([0, bins[-1]])

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('medium')
    ax.legend(loc='best', prop=fontP)

    # Plot vertical lines for unpaired peaks
    unpaired_peak_count = 0
    for _, peak in enumerate(signal1.unpaired_peaks):
        ax.axvline(signal1.freqs[peak], color=KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5)
        ax.annotate(
            f'Peak {unpaired_peak_count + 1}',
            (signal1.freqs[peak], t[-1] * 0.05),
            textcoords='data',
            color=KLIPPAIN_COLORS['red_pink'],
            rotation=90,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
        )
        unpaired_peak_count += 1

    for _, peak in enumerate(signal2.unpaired_peaks):
        ax.axvline(signal2.freqs[peak], color=KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5)
        ax.annotate(
            f'Peak {unpaired_peak_count + 1}',
            (signal2.freqs[peak], t[-1] * 0.05),
            textcoords='data',
            color=KLIPPAIN_COLORS['red_pink'],
            rotation=90,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
        )
        unpaired_peak_count += 1

    # Plot vertical lines and zones for paired peaks
    for idx, (peak1, peak2) in enumerate(signal1.paired_peaks):
        label = ALPHABET[idx]
        x_min = min(peak1[1], peak2[1])
        x_max = max(peak1[1], peak2[1])
        ax.axvline(x_min, color=KLIPPAIN_COLORS['dark_purple'], linestyle='dotted', linewidth=1.5)
        ax.axvline(x_max, color=KLIPPAIN_COLORS['dark_purple'], linestyle='dotted', linewidth=1.5)
        ax.fill_between([x_min, x_max], 0, np.max(combined_divergent), color=KLIPPAIN_COLORS['dark_purple'], alpha=0.3)
        ax.annotate(
            f'Peaks {label}',
            (x_min, t[-1] * 0.05),
            textcoords='data',
            color=KLIPPAIN_COLORS['dark_purple'],
            rotation=90,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
        )

    return


######################################################################
# Custom tools
######################################################################


# Original Klipper function to get the PSD data of a raw accelerometer signal
def compute_signal_data(data, max_freq):
    helper = shaper_calibrate.ShaperCalibrate(printer=None)
    calibration_data = helper.process_accelerometer_data(data)

    freqs = calibration_data.freq_bins[calibration_data.freq_bins <= max_freq]
    psd = calibration_data.get_psd('all')[calibration_data.freq_bins <= max_freq]

    _, peaks, _ = detect_peaks(psd, freqs, PEAKS_DETECTION_THRESHOLD * psd.max())

    return SignalData(freqs=freqs, psd=psd, peaks=peaks, paired_peaks=None, unpaired_peaks=None)


######################################################################
# Startup and main routines
######################################################################


def belts_calibration(lognames, klipperdir='~/klipper', max_freq=200.0, st_version=None):
    set_locale()
    global shaper_calibrate
    shaper_calibrate = setup_klipper_import(klipperdir)

    # Parse data
    datas = [parse_log(fn) for fn in lognames]
    if len(datas) > 2:
        raise ValueError('Incorrect number of .csv files used (this function needs exactly two files to compare them)!')

    # Compute calibration data for the two datasets with automatic peaks detection
    signal1 = compute_signal_data(datas[0], max_freq)
    signal2 = compute_signal_data(datas[1], max_freq)
    combined_sum, combined_divergent, bins, t = compute_combined_spectrogram(datas[0], datas[1])
    del datas

    # Pair the peaks across the two datasets
    paired_peaks, unpaired_peaks1, unpaired_peaks2 = pair_peaks(
        signal1.peaks, signal1.freqs, signal1.psd, signal2.peaks, signal2.freqs, signal2.psd
    )
    signal1 = signal1._replace(paired_peaks=paired_peaks, unpaired_peaks=unpaired_peaks1)
    signal2 = signal2._replace(paired_peaks=paired_peaks, unpaired_peaks=unpaired_peaks2)

    # Compute the similarity (using cross-correlation of the PSD signals)
    similarity_factor = compute_curve_similarity_factor(
        signal1.freqs, signal1.psd, signal2.freqs, signal2.psd, CURVE_SIMILARITY_SIGMOID_K
    )
    print_with_c_locale(f'Belts estimated similarity: {similarity_factor:.1f}%')
    # Compute the MHI value from the differential spectrogram sum of gradient, salted with the similarity factor and the number of
    # unpaired peaks from the belts frequency profile. Be careful, this value is highly opinionated and is pretty experimental!
    mhi, textual_mhi = compute_mhi(
        combined_sum, similarity_factor, len(signal1.unpaired_peaks) + len(signal2.unpaired_peaks)
    )
    print_with_c_locale(f'[experimental] Mechanical Health Indicator: {textual_mhi.lower()} ({mhi:.1f}%)')

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

    # Add title
    title_line1 = 'RELATIVE BELTS CALIBRATION TOOL'
    fig.text(
        0.12, 0.965, title_line1, ha='left', va='bottom', fontsize=20, color=KLIPPAIN_COLORS['purple'], weight='bold'
    )
    try:
        filename = lognames[0].split('/')[-1]
        dt = datetime.strptime(f"{filename.split('_')[1]} {filename.split('_')[2]}", '%Y%m%d %H%M%S')
        title_line2 = dt.strftime('%x %X')
    except Exception:
        print_with_c_locale(
            'Warning: CSV filenames look to be different than expected (%s , %s)' % (lognames[0], lognames[1])
        )
        title_line2 = lognames[0].split('/')[-1] + ' / ' + lognames[1].split('/')[-1]
    fig.text(0.12, 0.957, title_line2, ha='left', va='top', fontsize=16, color=KLIPPAIN_COLORS['dark_purple'])

    # Plot the graphs
    plot_compare_frequency(ax1, lognames, signal1, signal2, similarity_factor, max_freq)
    plot_difference_spectrogram(ax2, signal1, signal2, t, bins, combined_divergent, textual_mhi, max_freq)

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
    usage = '%prog [options] <raw logs>'
    opts = optparse.OptionParser(usage)
    opts.add_option('-o', '--output', type='string', dest='output', default=None, help='filename of output graph')
    opts.add_option('-f', '--max_freq', type='float', default=200.0, help='maximum frequency to graph')
    opts.add_option(
        '-k', '--klipper_dir', type='string', dest='klipperdir', default='~/klipper', help='main klipper directory'
    )
    options, args = opts.parse_args()
    if len(args) < 1:
        opts.error('Incorrect number of arguments')
    if options.output is None:
        opts.error('You must specify an output file.png to use the script (option -o)')

    fig = belts_calibration(args, options.klipperdir, options.max_freq)
    fig.savefig(options.output, dpi=150)


if __name__ == '__main__':
    main()
