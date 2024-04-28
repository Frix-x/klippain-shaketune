#!/usr/bin/env python3

##################################################
#### DIRECTIONAL VIBRATIONS PLOTTING SCRIPT ######
##################################################
# Written by Frix_x#0161 #

import math
import optparse
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib
import matplotlib.font_manager
import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

matplotlib.use('Agg')

from ..helpers.common_func import (
    compute_mechanical_parameters,
    detect_peaks,
    identify_low_energy_zones,
    parse_log,
    setup_klipper_import,
)
from ..helpers.locale_utils import print_with_c_locale, set_locale

PEAKS_DETECTION_THRESHOLD = 0.05
PEAKS_RELATIVE_HEIGHT_THRESHOLD = 0.04
CURVE_SIMILARITY_SIGMOID_K = 0.5
SPEEDS_VALLEY_DETECTION_THRESHOLD = 0.7  # Lower is more sensitive
SPEEDS_AROUND_PEAK_DELETION = 3  # to delete +-3mm/s around a peak
ANGLES_VALLEY_DETECTION_THRESHOLD = 1.1  # Lower is more sensitive

KLIPPAIN_COLORS = {
    'purple': '#70088C',
    'orange': '#FF8D32',
    'dark_purple': '#150140',
    'dark_orange': '#F24130',
    'red_pink': '#F2055C',
}


######################################################################
# Computation
######################################################################


# Call to the official Klipper input shaper object to do the PSD computation
def calc_freq_response(data):
    helper = shaper_calibrate.ShaperCalibrate(printer=None)
    return helper.process_accelerometer_data(data)


# Calculate motor frequency profiles based on the measured Power Spectral Density (PSD) measurements for the machine kinematics
# main angles and then create a global motor profile as a weighted average (from their own vibrations) of all calculated profiles
def compute_motor_profiles(freqs, psds, all_angles_energy, measured_angles=None, energy_amplification_factor=2):
    if measured_angles is None:
        measured_angles = [0, 90]

    motor_profiles = {}
    weighted_sum_profiles = np.zeros_like(freqs)
    total_weight = 0
    conv_filter = np.ones(20) / 20

    # Creating the PSD motor profiles for each angles
    for angle in measured_angles:
        # Calculate the sum of PSDs for the current angle and then convolve
        sum_curve = np.sum(np.array([psds[angle][speed] for speed in psds[angle]]), axis=0)
        motor_profiles[angle] = np.convolve(sum_curve / len(psds[angle]), conv_filter, mode='same')

        # Calculate weights
        angle_energy = (
            all_angles_energy[angle] ** energy_amplification_factor
        )  # First weighting factor is based on the total vibrations of the machine at the specified angle
        curve_area = (
            np.trapz(motor_profiles[angle], freqs) ** energy_amplification_factor
        )  # Additional weighting factor is based on the area under the current motor profile at this specified angle
        total_angle_weight = angle_energy * curve_area

        # Update weighted sum profiles to get the global motor profile
        weighted_sum_profiles += motor_profiles[angle] * total_angle_weight
        total_weight += total_angle_weight

    # Creating a global average motor profile that is the weighted average of all the PSD motor profiles
    global_motor_profile = weighted_sum_profiles / total_weight if total_weight != 0 else weighted_sum_profiles

    return motor_profiles, global_motor_profile


# Since it was discovered that there is no non-linear mixing in the stepper "steps" vibrations, instead of measuring
# the effects of each speeds at each angles, this function simplify it by using only the main motors axes (X/Y for Cartesian
# printers and A/B for CoreXY) measurements and project each points on the [0,360] degrees range using trigonometry
# to "sum" the vibration impact of each axis at every points of the generated spectrogram. The result is very similar at the end.
def compute_dir_speed_spectrogram(measured_speeds, data, kinematics='cartesian', measured_angles=None):
    if measured_angles is None:
        measured_angles = [0, 90]

    # We want to project the motor vibrations measured on their own axes on the [0, 360] range
    spectrum_angles = np.linspace(0, 360, 720)  # One point every 0.5 degrees
    spectrum_speeds = np.linspace(min(measured_speeds), max(measured_speeds), len(measured_speeds) * 6)
    spectrum_vibrations = np.zeros((len(spectrum_angles), len(spectrum_speeds)))

    def get_interpolated_vibrations(data, speed, speeds):
        idx = np.clip(np.searchsorted(speeds, speed, side='left'), 1, len(speeds) - 1)
        lower_speed = speeds[idx - 1]
        upper_speed = speeds[idx]
        lower_vibrations = data.get(lower_speed, 0)
        upper_vibrations = data.get(upper_speed, 0)
        return lower_vibrations + (speed - lower_speed) * (upper_vibrations - lower_vibrations) / (
            upper_speed - lower_speed
        )

    # Precompute trigonometric values and constant before the loop
    angle_radians = np.deg2rad(spectrum_angles)
    cos_vals = np.cos(angle_radians)
    sin_vals = np.sin(angle_radians)
    sqrt_2_inv = 1 / math.sqrt(2)

    # Compute the spectrum vibrations for each angle and speed combination
    for target_angle_idx, (cos_val, sin_val) in enumerate(zip(cos_vals, sin_vals)):
        for target_speed_idx, target_speed in enumerate(spectrum_speeds):
            if kinematics == 'cartesian':
                speed_1 = np.abs(target_speed * cos_val)
                speed_2 = np.abs(target_speed * sin_val)
            elif kinematics == 'corexy':
                speed_1 = np.abs(target_speed * (cos_val + sin_val) * sqrt_2_inv)
                speed_2 = np.abs(target_speed * (cos_val - sin_val) * sqrt_2_inv)

            vibrations_1 = get_interpolated_vibrations(data[measured_angles[0]], speed_1, measured_speeds)
            vibrations_2 = get_interpolated_vibrations(data[measured_angles[1]], speed_2, measured_speeds)
            spectrum_vibrations[target_angle_idx, target_speed_idx] = vibrations_1 + vibrations_2

    return spectrum_angles, spectrum_speeds, spectrum_vibrations


def compute_angle_powers(spectrogram_data):
    angles_powers = np.trapz(spectrogram_data, axis=1)

    # Since we want to plot it on a continuous polar plot later on, we need to append parts of
    # the array to start and end of it to smooth transitions when doing the convolution
    # and get the same value at modulo 360. Then we return the array without the extras
    extended_angles_powers = np.concatenate([angles_powers[-9:], angles_powers, angles_powers[:9]])
    convolved_extended = np.convolve(extended_angles_powers, np.ones(15) / 15, mode='same')

    return convolved_extended[9:-9]


def compute_speed_powers(spectrogram_data, smoothing_window=15):
    min_values = np.amin(spectrogram_data, axis=0)
    max_values = np.amax(spectrogram_data, axis=0)
    var_values = np.var(spectrogram_data, axis=0)

    # rescale the variance to the same range as max_values to plot it on the same graph
    var_values = var_values / var_values.max() * max_values.max()

    # Create a vibration metric that is the product of the max values and the variance to quantify the best
    # speeds that have at the same time a low global energy level that is also consistent at every angles
    vibration_metric = max_values * var_values

    # utility function to pad and smooth the data avoiding edge effects
    conv_filter = np.ones(smoothing_window) / smoothing_window
    window = int(smoothing_window / 2)

    def pad_and_smooth(data):
        data_padded = np.pad(data, (window,), mode='edge')
        smoothed_data = np.convolve(data_padded, conv_filter, mode='valid')
        return smoothed_data

    # Stack the arrays and apply padding and smoothing in batch
    data_arrays = np.stack([min_values, max_values, var_values, vibration_metric])
    smoothed_arrays = np.array([pad_and_smooth(data) for data in data_arrays])

    return smoothed_arrays


# Function that filter and split the good_speed ranges. The goal is to remove some zones around
# additional detected small peaks in order to suppress them if there is a peak, even if it's low,
# that's probably due to a crossing in the motor resonance pattern that still need to be removed
def filter_and_split_ranges(all_speeds, good_speeds, peak_speed_indices, deletion_range):
    # Process each range to filter out and split based on peak indices
    filtered_good_speeds = []
    for start, end, energy in good_speeds:
        start_speed, end_speed = all_speeds[start], all_speeds[end]
        # Identify peaks that intersect with the current speed range
        intersecting_peaks_indices = [
            idx for speed, idx in peak_speed_indices.items() if start_speed <= speed <= end_speed
        ]

        if not intersecting_peaks_indices:
            filtered_good_speeds.append((start, end, energy))
        else:
            intersecting_peaks_indices.sort()
            current_start = start

            for peak_index in intersecting_peaks_indices:
                before_peak_end = max(current_start, peak_index - deletion_range)
                if current_start < before_peak_end:
                    filtered_good_speeds.append((current_start, before_peak_end, energy))
                current_start = peak_index + deletion_range + 1

            if current_start < end:
                filtered_good_speeds.append((current_start, end, energy))

    # Sorting by start point once and then merge overlapping ranges
    sorted_ranges = sorted(filtered_good_speeds, key=lambda x: x[0])
    merged_ranges = [sorted_ranges[0]]

    for current in sorted_ranges[1:]:
        last_merged_start, last_merged_end, last_merged_energy = merged_ranges[-1]
        if current[0] <= last_merged_end:
            new_end = max(last_merged_end, current[1])
            new_energy = min(last_merged_energy, current[2])
            merged_ranges[-1] = (last_merged_start, new_end, new_energy)
        else:
            merged_ranges.append(current)

    return merged_ranges


# This function allow the computation of a symmetry score that reflect the spectrogram apparent symmetry between
# measured axes on both the shape of the signal and the energy level consistency across both side of the signal
def compute_symmetry_analysis(all_angles, spectrogram_data, measured_angles=None):
    if measured_angles is None:
        measured_angles = [0, 90]

    total_spectrogram_angles = len(all_angles)
    half_spectrogram_angles = total_spectrogram_angles // 2

    # Extend the spectrogram by adding half to the beginning (in order to not get an out of bounds error later)
    extended_spectrogram = np.concatenate((spectrogram_data[-half_spectrogram_angles:], spectrogram_data), axis=0)

    # Calculate the split index directly within the slicing
    midpoint_angle = np.mean(measured_angles)
    split_index = int(midpoint_angle * (total_spectrogram_angles / 360) + half_spectrogram_angles)
    half_segment_length = half_spectrogram_angles // 2

    # Slice out the two segments of the spectrogram and flatten them for comparison
    segment_1_flattened = extended_spectrogram[split_index - half_segment_length : split_index].flatten()
    segment_2_flattened = extended_spectrogram[split_index : split_index + half_segment_length].flatten()

    # Compute the correlation coefficient between the two segments of spectrogram
    correlation = np.corrcoef(segment_1_flattened, segment_2_flattened)[0, 1]
    percentage_correlation_biased = (100 * np.power(correlation, 0.75)) + 10

    return np.clip(0, 100, percentage_correlation_biased)


######################################################################
# Graphing
######################################################################


def plot_angle_profile_polar(ax, angles, angles_powers, low_energy_zones, symmetry_factor):
    angles_radians = np.deg2rad(angles)

    ax.set_title('Polar angle energy profile', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    ax.plot(angles_radians, angles_powers, color=KLIPPAIN_COLORS['purple'], zorder=5)
    ax.fill(angles_radians, angles_powers, color=KLIPPAIN_COLORS['purple'], alpha=0.3)
    ax.set_xlim([0, np.deg2rad(360)])
    ymax = angles_powers.max() * 1.05
    ax.set_ylim([0, ymax])
    ax.set_thetagrids([theta * 15 for theta in range(360 // 15)])

    ax.text(
        0,
        0,
        f'Symmetry: {symmetry_factor:.1f}%',
        ha='center',
        va='center',
        color=KLIPPAIN_COLORS['red_pink'],
        fontsize=12,
        fontweight='bold',
        zorder=6,
    )

    for _, (start, end, _) in enumerate(low_energy_zones):
        ax.axvline(
            angles_radians[start],
            angles_powers[start] / ymax,
            color=KLIPPAIN_COLORS['red_pink'],
            linestyle='dotted',
            linewidth=1.5,
        )
        ax.axvline(
            angles_radians[end],
            angles_powers[end] / ymax,
            color=KLIPPAIN_COLORS['red_pink'],
            linestyle='dotted',
            linewidth=1.5,
        )
        ax.fill_between(
            angles_radians[start:end], angles_powers[start:end], angles_powers.max() * 1.05, color='green', alpha=0.2
        )

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    # Polar plot doesn't follow the gridspec margin, so we adjust it manually here
    pos = ax.get_position()
    new_pos = [pos.x0 - 0.01, pos.y0 - 0.01, pos.width, pos.height]
    ax.set_position(new_pos)

    return


def plot_global_speed_profile(
    ax,
    all_speeds,
    sp_min_energy,
    sp_max_energy,
    sp_variance_energy,
    vibration_metric,
    num_peaks,
    peaks,
    low_energy_zones,
):
    ax.set_title('Global speed energy profile', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('Energy')
    ax2 = ax.twinx()
    ax2.yaxis.set_visible(False)

    ax.plot(all_speeds, sp_min_energy, label='Minimum', color=KLIPPAIN_COLORS['dark_purple'], zorder=5)
    ax.plot(all_speeds, sp_max_energy, label='Maximum', color=KLIPPAIN_COLORS['purple'], zorder=5)
    ax.plot(all_speeds, sp_variance_energy, label='Variance', color=KLIPPAIN_COLORS['orange'], zorder=5, linestyle='--')
    ax2.plot(
        all_speeds,
        vibration_metric,
        label=f'Vibration metric ({num_peaks} bad peaks)',
        color=KLIPPAIN_COLORS['red_pink'],
        zorder=5,
    )

    ax.set_xlim([all_speeds.min(), all_speeds.max()])
    ax.set_ylim([0, sp_max_energy.max() * 1.15])

    y2min = -(vibration_metric.max() * 0.025)
    y2max = vibration_metric.max() * 1.07
    ax2.set_ylim([y2min, y2max])

    if peaks is not None and len(peaks) > 0:
        ax2.plot(all_speeds[peaks], vibration_metric[peaks], 'x', color='black', markersize=8, zorder=10)
        for idx, peak in enumerate(peaks):
            ax2.annotate(
                f'{idx+1}',
                (all_speeds[peak], vibration_metric[peak]),
                textcoords='offset points',
                xytext=(5, 5),
                fontweight='bold',
                ha='left',
                fontsize=13,
                color=KLIPPAIN_COLORS['red_pink'],
                zorder=10,
            )

    for idx, (start, end, _) in enumerate(low_energy_zones):
        # ax2.axvline(all_speeds[start], color=KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5, zorder=8)
        # ax2.axvline(all_speeds[end], color=KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5, zorder=8)
        ax2.fill_between(
            all_speeds[start:end],
            y2min,
            vibration_metric[start:end],
            color='green',
            alpha=0.2,
            label=f'Zone {idx+1}: {all_speeds[start]:.1f} to {all_speeds[end]:.1f} mm/s',
        )

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper left', prop=fontP)
    ax2.legend(loc='upper right', prop=fontP)

    return


def plot_angular_speed_profiles(ax, speeds, angles, spectrogram_data, kinematics='cartesian'):
    ax.set_title('Angular speed energy profiles', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('Energy')

    # Define mappings for labels and colors to simplify plotting commands
    angle_settings = {
        0: ('X (0 deg)', 'purple', 10),
        90: ('Y (90 deg)', 'dark_purple', 5),
        45: ('A (45 deg)' if kinematics == 'corexy' else '45 deg', 'orange', 10),
        135: ('B (135 deg)' if kinematics == 'corexy' else '135 deg', 'dark_orange', 5),
    }

    # Plot each angle using settings from the dictionary
    for angle, (label, color, zorder) in angle_settings.items():
        idx = np.searchsorted(angles, angle, side='left')
        ax.plot(speeds, spectrogram_data[idx], label=label, color=KLIPPAIN_COLORS[color], zorder=zorder)

    ax.set_xlim([speeds.min(), speeds.max()])
    max_value = max(spectrogram_data[angle].max() for angle in [0, 45, 90, 135])
    ax.set_ylim([0, max_value * 1.1])

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper right', prop=fontP)

    return


def plot_motor_profiles(ax, freqs, main_angles, motor_profiles, global_motor_profile, max_freq):
    ax.set_title('Motor frequency profile', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_ylabel('Energy')
    ax.set_xlabel('Frequency (Hz)')

    ax2 = ax.twinx()
    ax2.yaxis.set_visible(False)

    # Global weighted average motor profile
    ax.plot(freqs, global_motor_profile, label='Combined', color=KLIPPAIN_COLORS['purple'], zorder=5)
    max_value = global_motor_profile.max()

    # Mapping of angles to axis names
    angle_settings = {0: 'X', 90: 'Y', 45: 'A', 135: 'B'}

    # And then plot the motor profiles at each measured angles
    for angle in main_angles:
        profile_max = motor_profiles[angle].max()
        if profile_max > max_value:
            max_value = profile_max
        label = f'{angle_settings[angle]} ({angle} deg)' if angle in angle_settings else f'{angle} deg'
        ax.plot(freqs, motor_profiles[angle], linestyle='--', label=label, zorder=2)

    ax.set_xlim([0, max_freq])
    ax.set_ylim([0, max_value * 1.1])
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Then add the motor resonance peak to the graph and print some infos about it
    motor_fr, motor_zeta, motor_res_idx, lowfreq_max = compute_mechanical_parameters(global_motor_profile, freqs, 30)
    if lowfreq_max:
        print_with_c_locale(
            '[WARNING] There are a lot of low frequency vibrations that can alter the readings. This is probably due to the test being performed at too high an acceleration!'
        )
        print_with_c_locale(
            'Try lowering the ACCEL value and/or increasing the SIZE value before restarting the macro to ensure that only constant speeds are being recorded and that the dynamic behavior of the machine is not affecting the measurements'
        )
    if motor_zeta is not None:
        print_with_c_locale(
            'Motors have a main resonant frequency at %.1fHz with an estimated damping ratio of %.3f'
            % (motor_fr, motor_zeta)
        )
    else:
        print_with_c_locale(
            'Motors have a main resonant frequency at %.1fHz but it was impossible to estimate a damping ratio.'
            % (motor_fr)
        )

    ax.plot(freqs[motor_res_idx], global_motor_profile[motor_res_idx], 'x', color='black', markersize=10)
    ax.annotate(
        'R',
        (freqs[motor_res_idx], global_motor_profile[motor_res_idx]),
        textcoords='offset points',
        xytext=(15, 5),
        ha='right',
        fontsize=14,
        color=KLIPPAIN_COLORS['red_pink'],
        weight='bold',
    )

    ax2.plot([], [], ' ', label='Motor resonant frequency (ω0): %.1fHz' % (motor_fr))
    if motor_zeta is not None:
        ax2.plot([], [], ' ', label='Motor damping ratio (ζ): %.3f' % (motor_zeta))
    else:
        ax2.plot([], [], ' ', label='No damping ratio computed')

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper left', prop=fontP)
    ax2.legend(loc='upper right', prop=fontP)

    return


def plot_vibration_spectrogram_polar(ax, angles, speeds, spectrogram_data):
    angles_radians = np.radians(angles)

    # Assuming speeds defines the radial distance from the center, we need to create a meshgrid
    # for both angles and speeds to map the spectrogram data onto a polar plot correctly
    radius, theta = np.meshgrid(speeds, angles_radians)

    ax.set_title(
        'Polar vibrations heatmap', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold', va='bottom'
    )
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    ax.pcolormesh(theta, radius, spectrogram_data, norm=matplotlib.colors.LogNorm(), cmap='inferno', shading='auto')
    ax.set_thetagrids([theta * 15 for theta in range(360 // 15)])
    ax.tick_params(axis='y', which='both', colors='white', labelsize='medium')
    ax.set_ylim([0, max(speeds)])

    # Polar plot doesn't follow the gridspec margin, so we adjust it manually here
    pos = ax.get_position()
    new_pos = [pos.x0 - 0.01, pos.y0 - 0.01, pos.width, pos.height]
    ax.set_position(new_pos)

    return


def plot_vibration_spectrogram(ax, angles, speeds, spectrogram_data, peaks):
    ax.set_title('Vibrations heatmap', fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('Angle (deg)')

    ax.imshow(
        spectrogram_data,
        norm=matplotlib.colors.LogNorm(),
        cmap='inferno',
        aspect='auto',
        extent=[speeds[0], speeds[-1], angles[0], angles[-1]],
        origin='lower',
        interpolation='antialiased',
    )

    # Add peaks lines in the spectrogram to get hint from peaks found in the first graph
    if peaks is not None and len(peaks) > 0:
        for idx, peak in enumerate(peaks):
            ax.axvline(speeds[peak], color='cyan', linewidth=0.75)
            ax.annotate(
                f'Peak {idx+1}',
                (speeds[peak], angles[-1] * 0.9),
                textcoords='data',
                color='cyan',
                rotation=90,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
            )

    return


def plot_motor_config_txt(fig, motors, differences):
    motor_details = [(motors[0], 'X motor'), (motors[1], 'Y motor')]

    distance = 0.12
    if motors[0].get_property('autotune_enabled'):
        distance = 0.24
        config_blocks = [
            f"| {lbl}: {mot.get_property('motor').upper()} on {mot.get_property('tmc').upper()} @ {mot.get_property('voltage')}V {mot.get_property('run_current')}A"
            for mot, lbl in motor_details
        ]
        config_blocks.append('| TMC Autotune enabled')
    else:
        config_blocks = [
            f"| {lbl}: {mot.get_property('tmc').upper()} @ {mot.get_property('run_current')}A"
            for mot, lbl in motor_details
        ]
        config_blocks.append('| TMC Autotune not detected')

    for idx, block in enumerate(config_blocks):
        fig.text(
            0.40, 0.990 - 0.015 * idx, block, ha='left', va='top', fontsize=10, color=KLIPPAIN_COLORS['dark_purple']
        )

    tmc_registers = motors[0].get_registers()
    idx = -1
    for idx, (register, settings) in enumerate(tmc_registers.items()):
        settings_str = ' '.join(f'{k}={v}' for k, v in settings.items())
        tmc_block = f'| {register.upper()}: {settings_str}'
        fig.text(
            0.40 + distance,
            0.990 - 0.015 * idx,
            tmc_block,
            ha='left',
            va='top',
            fontsize=10,
            color=KLIPPAIN_COLORS['dark_purple'],
        )

    if differences is not None:
        differences_text = f'| Y motor diff: {differences}'
        fig.text(
            0.40 + distance,
            0.990 - 0.015 * (idx + 1),
            differences_text,
            ha='left',
            va='top',
            fontsize=10,
            color=KLIPPAIN_COLORS['dark_purple'],
        )


######################################################################
# Startup and main routines
######################################################################


def extract_angle_and_speed(logname):
    try:
        match = re.search(r'an(\d+)_\d+sp(\d+)_\d+', os.path.basename(logname))
        if match:
            angle = match.group(1)
            speed = match.group(2)
        else:
            raise ValueError(f'File {logname} does not match expected format. Clean your /tmp folder and start again!')
    except AttributeError as err:
        raise ValueError(
            f'File {logname} does not match expected format. Clean your /tmp folder and start again!'
        ) from err
    return float(angle), float(speed)


def vibrations_profile(
    lognames, klipperdir='~/klipper', kinematics='cartesian', accel=None, max_freq=1000.0, st_version=None, motors=None
):
    set_locale()
    global shaper_calibrate
    shaper_calibrate = setup_klipper_import(klipperdir)

    if kinematics == 'cartesian':
        main_angles = [0, 90]
    elif kinematics == 'corexy':
        main_angles = [45, 135]
    else:
        raise ValueError('Only Cartesian and CoreXY kinematics are supported by this tool at the moment!')

    psds = defaultdict(lambda: defaultdict(list))
    psds_sum = defaultdict(lambda: defaultdict(list))
    target_freqs_initialized = False

    for logname in lognames:
        data = parse_log(logname)
        angle, speed = extract_angle_and_speed(logname)
        freq_response = calc_freq_response(data)
        first_freqs = freq_response.freq_bins
        psd_sum = freq_response.psd_sum

        if not target_freqs_initialized:
            target_freqs = first_freqs[first_freqs <= max_freq]
            target_freqs_initialized = True

        psd_sum = psd_sum[first_freqs <= max_freq]
        first_freqs = first_freqs[first_freqs <= max_freq]

        # Store the interpolated PSD and integral values
        psds[angle][speed] = np.interp(target_freqs, first_freqs, psd_sum)
        psds_sum[angle][speed] = np.trapz(psd_sum, first_freqs)

    measured_angles = sorted(psds_sum.keys())
    measured_speeds = sorted({speed for angle_speeds in psds_sum.values() for speed in angle_speeds.keys()})

    for main_angle in main_angles:
        if main_angle not in measured_angles:
            raise ValueError('Measurements not taken at the correct angles for the specified kinematics!')

    # Precompute the variables used in plot functions
    all_angles, all_speeds, spectrogram_data = compute_dir_speed_spectrogram(
        measured_speeds, psds_sum, kinematics, main_angles
    )
    all_angles_energy = compute_angle_powers(spectrogram_data)
    sp_min_energy, sp_max_energy, sp_variance_energy, vibration_metric = compute_speed_powers(spectrogram_data)
    motor_profiles, global_motor_profile = compute_motor_profiles(target_freqs, psds, all_angles_energy, main_angles)

    # symmetry_factor = compute_symmetry_analysis(all_angles, all_angles_energy)
    symmetry_factor = compute_symmetry_analysis(all_angles, spectrogram_data, main_angles)
    print_with_c_locale(f'Machine estimated vibration symmetry: {symmetry_factor:.1f}%')

    # Analyze low variance ranges of vibration energy across all angles for each speed to identify clean speeds
    # and highlight them. Also find the peaks to identify speeds to avoid due to high resonances
    num_peaks, vibration_peaks, peaks_speeds = detect_peaks(
        vibration_metric,
        all_speeds,
        PEAKS_DETECTION_THRESHOLD * vibration_metric.max(),
        PEAKS_RELATIVE_HEIGHT_THRESHOLD,
        10,
        10,
    )
    formated_peaks_speeds = ['{:.1f}'.format(pspeed) for pspeed in peaks_speeds]
    print_with_c_locale(
        'Vibrations peaks detected: %d @ %s mm/s (avoid setting a speed near these values in your slicer print profile)'
        % (num_peaks, ', '.join(map(str, formated_peaks_speeds)))
    )

    good_speeds = identify_low_energy_zones(vibration_metric, SPEEDS_VALLEY_DETECTION_THRESHOLD)
    if good_speeds is not None:
        deletion_range = int(SPEEDS_AROUND_PEAK_DELETION / (all_speeds[1] - all_speeds[0]))
        peak_speed_indices = {pspeed: np.where(all_speeds == pspeed)[0][0] for pspeed in set(peaks_speeds)}

        # Filter and split ranges based on peak indices, avoiding overlaps
        good_speeds = filter_and_split_ranges(all_speeds, good_speeds, peak_speed_indices, deletion_range)

        # Add some logging about the good speeds found
        print_with_c_locale(f'Lowest vibrations speeds ({len(good_speeds)} ranges sorted from best to worse):')
        for idx, (start, end, _) in enumerate(good_speeds):
            print_with_c_locale(f'{idx+1}: {all_speeds[start]:.1f} to {all_speeds[end]:.1f} mm/s')

    # Angle low energy valleys identification (good angles ranges) and print them to the console
    good_angles = identify_low_energy_zones(all_angles_energy, ANGLES_VALLEY_DETECTION_THRESHOLD)
    if good_angles is not None:
        print_with_c_locale(f'Lowest vibrations angles ({len(good_angles)} ranges sorted from best to worse):')
        for idx, (start, end, energy) in enumerate(good_angles):
            print_with_c_locale(
                f'{idx+1}: {all_angles[start]:.1f}° to {all_angles[end]:.1f}° (mean vibrations energy: {energy:.2f}% of max)'
            )

    # Create graph layout
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
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

    # Transform ax3 and ax4 to polar plots
    ax1.remove()
    ax1 = fig.add_subplot(2, 3, 1, projection='polar')
    ax4.remove()
    ax4 = fig.add_subplot(2, 3, 4, projection='polar')

    # Set the global .png figure size
    fig.set_size_inches(20, 11.5)

    # Add title
    title_line1 = 'MACHINE VIBRATIONS ANALYSIS TOOL'
    fig.text(
        0.060, 0.965, title_line1, ha='left', va='bottom', fontsize=20, color=KLIPPAIN_COLORS['purple'], weight='bold'
    )
    try:
        filename_parts = (lognames[0].split('/')[-1]).split('_')
        dt = datetime.strptime(f"{filename_parts[1]} {filename_parts[2].split('-')[0]}", '%Y%m%d %H%M%S')
        title_line2 = dt.strftime('%x %X')
        if accel is not None:
            title_line2 += ' at ' + str(accel) + ' mm/s² -- ' + kinematics.upper() + ' kinematics'
    except Exception:
        print_with_c_locale('Warning: CSV filenames appear to be different than expected (%s)' % (lognames[0]))
        title_line2 = lognames[0].split('/')[-1]
    fig.text(0.060, 0.957, title_line2, ha='left', va='top', fontsize=16, color=KLIPPAIN_COLORS['dark_purple'])

    # Add the motors infos to the top of the graph
    if motors is not None and len(motors) == 2:
        differences = motors[0].compare_to(motors[1])
        plot_motor_config_txt(fig, motors, differences)
        if differences is not None and kinematics == 'corexy':
            print_with_c_locale(f'Warning: motors have different TMC configurations!\n{differences}')

    # Plot the graphs
    plot_angle_profile_polar(ax1, all_angles, all_angles_energy, good_angles, symmetry_factor)
    plot_vibration_spectrogram_polar(ax4, all_angles, all_speeds, spectrogram_data)

    plot_global_speed_profile(
        ax2,
        all_speeds,
        sp_min_energy,
        sp_max_energy,
        sp_variance_energy,
        vibration_metric,
        num_peaks,
        vibration_peaks,
        good_speeds,
    )
    plot_angular_speed_profiles(ax3, all_speeds, all_angles, spectrogram_data, kinematics)
    plot_vibration_spectrogram(ax5, all_angles, all_speeds, spectrogram_data, vibration_peaks)

    plot_motor_profiles(ax6, target_freqs, main_angles, motor_profiles, global_motor_profile, max_freq)

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
    usage = '%prog [options] <raw logs>'
    opts = optparse.OptionParser(usage)
    opts.add_option('-o', '--output', type='string', dest='output', default=None, help='filename of output graph')
    opts.add_option(
        '-c', '--accel', type='int', dest='accel', default=None, help='accel value to be printed on the graph'
    )
    opts.add_option('-f', '--max_freq', type='float', default=1000.0, help='maximum frequency to graph')
    opts.add_option(
        '-k', '--klipper_dir', type='string', dest='klipperdir', default='~/klipper', help='main klipper directory'
    )
    opts.add_option(
        '-m',
        '--kinematics',
        type='string',
        dest='kinematics',
        default='cartesian',
        help='machine kinematics configuration',
    )
    options, args = opts.parse_args()
    if len(args) < 1:
        opts.error('No CSV file(s) to analyse')
    if options.output is None:
        opts.error('You must specify an output file.png to use the script (option -o)')
    if options.kinematics not in ['cartesian', 'corexy']:
        opts.error('Only cartesian and corexy kinematics are supported by this tool at the moment!')

    fig = vibrations_profile(args, options.klipperdir, options.kinematics, options.accel, options.max_freq)
    fig.savefig(options.output, dpi=150)


if __name__ == '__main__':
    main()
