#!/usr/bin/env python3

##################################################
#### DIRECTIONAL VIBRATIONS PLOTTING SCRIPT ######
##################################################
# Written by Frix_x#0161 #

# Be sure to make this script executable using SSH: type 'chmod +x ./graph_dir_vibrations.py' when in the folder !

#####################################################################
################ !!! DO NOT EDIT BELOW THIS LINE !!! ################
#####################################################################

import math
import optparse, matplotlib, re, os
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager, matplotlib.ticker, matplotlib.gridspec


matplotlib.use('Agg')

from locale_utils import set_locale, print_with_c_locale
from common_func import get_git_version, parse_log, setup_klipper_import, identify_low_energy_zones, compute_curve_similarity_factor, compute_mechanical_parameters, detect_peaks


PEAKS_DETECTION_THRESHOLD = 0.05
PEAKS_RELATIVE_HEIGHT_THRESHOLD = 0.04
CURVE_SIMILARITY_SIGMOID_K = 0.5
SPEEDS_VALLEY_DETECTION_THRESHOLD = 0.7 # Lower is more sensitive
ANGLES_VALLEY_DETECTION_THRESHOLD = 1.1 # Lower is more sensitive

KLIPPAIN_COLORS = {
    "purple": "#70088C",
    "orange": "#FF8D32",
    "dark_purple": "#150140",
    "dark_orange": "#F24130",
    "red_pink": "#F2055C"
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
def compute_motor_profiles(freqs, psds, all_angles_energy, measured_angles=[0, 90], energy_amplification_factor=2):
    motor_profiles = {}
    weighted_sum_profiles = np.zeros_like(freqs)
    total_weight = 0

    # Creating the PSD motor profiles for each angles
    for angle in measured_angles:
        sum_curve = np.zeros_like(freqs)
        for speed in psds[angle]:
            sum_curve += psds[angle][speed]

        motor_profiles[angle] = np.convolve(sum_curve / len(psds[angle]), np.ones(20)/20, mode='same')
        angle_energy = all_angles_energy[angle] ** energy_amplification_factor # First weighting factor based on the total vibrations of the machine at the specified angle
        curve_area = np.trapz(motor_profiles[angle], freqs) ** energy_amplification_factor # Additional weighting factor based on the area under the current motor profile at this specified angle
        total_angle_weight = angle_energy * curve_area
    
        weighted_sum_profiles += motor_profiles[angle] * total_angle_weight
        total_weight += total_angle_weight

    # Creating a global average motor profile that is the weighted average of all the PSD motor profiles
    global_motor_profile = weighted_sum_profiles / total_weight if total_weight != 0 else weighted_sum_profiles

    # return motor_profiles, np.convolve(global_motor_profile, np.ones(15)/15, mode='same')
    return motor_profiles, global_motor_profile


# Since it was discovered that there is no non-linear mixing in the stepper "steps" vibrations, instead of measuring
# the effects of each speeds at each angles, this function simplify it by using only the main motors axes (X/Y for Cartesian
# printers and A/B for CoreXY) measurements and project each points on the [0,360] degrees range using trigonometry
# to "sum" the vibration impact of each axis at every points of the generated spectrogram. The result is very similar at the end.
def compute_dir_speed_spectrogram(measured_speeds, data, kinematics="cartesian", measured_angles=[0, 90]):
    # We want to project the motor vibrations measured on their own axes on the [0, 360] range
    spectrum_angles = np.linspace(0, 360, 720) # One point every 0.5 degrees
    spectrum_speeds = np.linspace(min(measured_speeds), max(measured_speeds), len(measured_speeds) * 5) # 5 points between each speed measurements
    spectrum_vibrations = np.zeros((len(spectrum_angles), len(spectrum_speeds)))

    def get_interpolated_vibrations(data, speed, speeds):
        idx = np.searchsorted(speeds, speed, side="left")
        if idx == 0: return data[speeds[0]]
        if idx == len(speeds): return data[speeds[-1]]
        lower_speed = speeds[idx - 1]
        upper_speed = speeds[idx]
        lower_vibrations = data.get(lower_speed, 0)
        upper_vibrations = data.get(upper_speed, 0)
        interpolated_vibrations = lower_vibrations + (speed - lower_speed) * (upper_vibrations - lower_vibrations) / (upper_speed - lower_speed)
        return interpolated_vibrations

    for target_angle_idx, target_angle in enumerate(spectrum_angles):
        target_angle_rad = np.deg2rad(target_angle)
        for target_speed_idx, target_speed in enumerate(spectrum_speeds):
            if kinematics == "cartesian":
                speed_1 = np.abs(target_speed * np.cos(target_angle_rad))
                speed_2 = np.abs(target_speed * np.sin(target_angle_rad))
            elif kinematics == "corexy":
                speed_1 = np.abs(target_speed * (np.cos(target_angle_rad) + np.sin(target_angle_rad)) / math.sqrt(2))
                speed_2 = np.abs(target_speed * (np.cos(target_angle_rad) - np.sin(target_angle_rad)) / math.sqrt(2))

            vibrations_1 = get_interpolated_vibrations(data[measured_angles[0]], speed_1, measured_speeds)
            vibrations_2 = get_interpolated_vibrations(data[measured_angles[1]], speed_2, measured_speeds)
            spectrum_vibrations[target_angle_idx, target_speed_idx] = vibrations_1 + vibrations_2

    return spectrum_angles, spectrum_speeds, spectrum_vibrations


def compute_angle_powers(spectrogram_data):
    angles_powers = np.trapz(spectrogram_data, axis=1)

    # Since we want to plot it on a continuous polar plot later on, we need to append parts of
    # the array to start and end of it to smooth transitions when doing the convolution
    # and get the same value at modulo 360. Then we return the array without the extras
    extra_start = angles_powers[-9:]
    extra_end = angles_powers[:9]
    extended_angles_powers = np.concatenate([extra_start, angles_powers, extra_end])
    convolved_extended = np.convolve(extended_angles_powers, np.ones(15)/15, mode='same')

    return convolved_extended[9:-9]


def compute_speed_powers(spectrogram_data):
    min_values = np.amin(spectrogram_data, axis=0)
    max_values = np.amax(spectrogram_data, axis=0)
    avg_values = np.mean(spectrogram_data, axis=0)
    energy_variance = np.var(spectrogram_data, axis=0)

    min_values_smooth = np.convolve(min_values, np.ones(15)/15, mode='same')
    max_values_smooth = np.convolve(max_values, np.ones(15)/15, mode='same')
    avg_values_smooth = np.convolve(avg_values, np.ones(15)/15, mode='same')
    energy_variance_smooth = np.convolve(energy_variance, np.ones(15)/15, mode='same')

    return min_values_smooth, max_values_smooth, avg_values_smooth, energy_variance_smooth


# This function uses a nuanced approach to allow the computation of a score that reflect both the shape
# similarity of a signal (via cross-correlation) and the energy level consistency across the signal
def compute_symmetry_analysis(all_angles, angles_energy):
    # Split the signal in half
    first_half_indices = (0 <= all_angles) & (all_angles < 90)
    second_half_indices = (90 <= all_angles) & (all_angles < 180)
    x1, y1 = all_angles[first_half_indices], angles_energy[first_half_indices]
    x2, y2 = all_angles[second_half_indices], angles_energy[second_half_indices]

    # Reverse the second signal to compare them on a real symmetry
    x2, y2 = x2[::-1], y2[::-1]

    # Compute the similarity (using cross-correlation of the signals)
    similarity_factor = compute_curve_similarity_factor(x1, y1, x2, y2, CURVE_SIMILARITY_SIGMOID_K)

    # Because the signal of both half have approximately the same shape, this is not enough and we need to
    # add the total energy of each side in the equation to help discriminate differences in the symmetry
    energy_first_half = np.sum(y1**2)
    energy_second_half = np.sum(y2**2)
    energy_gap = np.abs(energy_first_half/energy_second_half - 1)

    # Compute an adjustement factor where close energies slightly increase the score and farther energies decrease the score
    if energy_gap <= 0.1: adjustment_factor = 1 + energy_gap
    else: adjustment_factor = 1 / (1 + 3 * (energy_gap - 0.1))

    # Adjust the similarity factor with the energy disparity
    adjusted_similarity_factor = similarity_factor * adjustment_factor

    return np.clip(adjusted_similarity_factor, 0, 100)

######################################################################
# Graphing
######################################################################

def plot_angle_profile_polar(ax, angles, angles_powers, low_energy_zones, symmetry_factor):
    angles_radians = np.deg2rad(angles)

    ax.set_title("Polar angle energy profile", fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    ax.plot(angles_radians, angles_powers, color=KLIPPAIN_COLORS['purple'], zorder=5)
    ax.fill(angles_radians, angles_powers, color=KLIPPAIN_COLORS['purple'], alpha=0.3)
    ax.set_xlim([0, np.deg2rad(360)])
    ymax = angles_powers.max() * 1.05
    ax.set_ylim([0, ymax])
    ax.set_thetagrids([theta * 15 for theta in range(360//15)])

    ax.text(0, 0, f'Symmetry: {symmetry_factor:.1f}%', ha='center', va='center', color=KLIPPAIN_COLORS['red_pink'], fontsize=12, fontweight='bold', zorder=6)

    for _, (start, end, _) in enumerate(low_energy_zones):
        ax.axvline(angles_radians[start], angles_powers[start]/ymax, color=KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5)
        ax.axvline(angles_radians[end], angles_powers[end]/ymax, color=KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5)
        ax.fill_between(angles_radians[start:end], angles_powers[start:end], angles_powers.max() * 1.05, color='green', alpha=0.3)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    # Polar plot doesn't follow the gridspec margin, so we adjust it manually here
    pos = ax.get_position()
    new_pos = [pos.x0 - 0.005, pos.y0, pos.width * 0.98, pos.height * 0.98]
    ax.set_position(new_pos)

    return

def plot_angle_profile(ax, angles, angles_powers, low_energy_zones):
    ax.set_title("Angle energy profile", fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_xlabel('Energy')
    ax.set_ylabel('Angle (deg)')

    ax.plot(angles_powers, angles, color=KLIPPAIN_COLORS['purple'], zorder=5)
    xmax = angles_powers.max() * 1.1
    ax.set_xlim([0, xmax])
    ax.set_ylim([angles.min(), angles.max()])

    for _, (start, end, _) in enumerate(low_energy_zones):
        ax.axhline(angles[start], 0, angles_powers[start]/xmax, color=KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5)
        ax.axhline(angles[end], 0, angles_powers[end]/xmax, color=KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5)
        ax.fill_betweenx(angles[start:end], 0, angles_powers[start:end], color='green', alpha=0.3)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    return

def plot_speed_profile(ax, all_speeds, sp_min_energy, sp_max_energy, sp_avg_energy, sp_energy_variance, num_peaks, peaks, low_energy_zones):
    ax.set_title("Speed energy profile", fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('Energy')
    ax2 = ax.twinx()
    ax2.yaxis.set_visible(False)

    ax.plot(all_speeds, sp_avg_energy, label='Average energy', color=KLIPPAIN_COLORS['dark_orange'], zorder=5)
    ax.plot(all_speeds, sp_min_energy, label='Minimum energy', color=KLIPPAIN_COLORS['dark_purple'], zorder=5)
    ax.plot(all_speeds, sp_max_energy, label='Maximum energy', color=KLIPPAIN_COLORS['purple'], zorder=5)
    ax2.plot(all_speeds, sp_energy_variance, label=f'Energy variance ({num_peaks} peaks)', color=KLIPPAIN_COLORS['orange'], zorder=5)

    ax.set_xlim([all_speeds.min(), all_speeds.max()])
    ax.set_ylim([0, sp_max_energy.max() * 1.1])
    ymax = sp_energy_variance.max() * 1.1
    ax2.set_ylim([0, ymax])

    if peaks is not None:
        ax2.plot(all_speeds[peaks], sp_energy_variance[peaks], "x", color='black', markersize=8, zorder=10)
        for idx, peak in enumerate(peaks):
            ax2.annotate(f"{idx+1}", (all_speeds[peak], sp_energy_variance[peak]),
                        textcoords="offset points", xytext=(8, 5), fontweight='bold',
                        ha='left', fontsize=13, color=KLIPPAIN_COLORS['red_pink'], zorder=10)

    for idx, (start, end, _) in enumerate(low_energy_zones):
        ax2.axvline(all_speeds[start], 0, sp_energy_variance[start]/ymax, color=KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5)
        ax2.axvline(all_speeds[end], 0, sp_energy_variance[start]/ymax, color=KLIPPAIN_COLORS['red_pink'], linestyle='dotted', linewidth=1.5)
        ax2.fill_between(all_speeds[start:end], 0, sp_energy_variance[start:end], color='green', alpha=0.3, label=f'Zone {idx+1}: {all_speeds[start]:.1f} to {all_speeds[end]:.1f} mm/s')

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper left', prop=fontP)
    ax2.legend(loc='upper right', prop=fontP)

    return

def plot_motor_profiles(ax, freqs, main_angles, motor_profiles, global_motor_profile):
    ax.set_title("Motor frequency profile", fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_ylabel('Energy')
    ax.set_xlabel('Frequency (Hz)')

    # Global weighted average motor profile
    ax.plot(freqs, global_motor_profile, label="Combined profile", color=KLIPPAIN_COLORS['purple'], zorder=5)
    max_value = global_motor_profile.max()

    # And then plot the motor profiles at each measured angles
    for angle in main_angles:
        profile_max = motor_profiles[angle].max()
        if profile_max > max_value:
            max_value = profile_max
        ax.plot(freqs, motor_profiles[angle], linestyle='--', label=f'{angle} deg', zorder=2)

    ax.set_xlim([0, 400])
    ax.set_ylim([0, max_value * 1.1])

    # Then add the motor resonance peak to the graph and print some infos about it
    motor_fr, motor_zeta, motor_res_idx = compute_mechanical_parameters(global_motor_profile, freqs)
    if motor_fr > 25:
        print_with_c_locale("Motors have a main resonant frequency at %.1fHz with an estimated damping ratio of %.3f" % (motor_fr, motor_zeta))
    else:
        print_with_c_locale("The detected resonance frequency of the motors is too low (%.1fHz). This is probably due to the test run with too high acceleration!" % motor_fr)
        print_with_c_locale("Try lowering the ACCEL value before restarting the macro to ensure that only constant speeds are recorded and that the dynamic behavior of the machine is not impacting the measurements.")

    ax.plot(freqs[motor_res_idx], global_motor_profile[motor_res_idx], "x", color='black', markersize=8)
    ax.annotate(f"R", (freqs[motor_res_idx], global_motor_profile[motor_res_idx]), 
                textcoords="offset points", xytext=(10, 5), 
                ha='right', fontsize=13, color=KLIPPAIN_COLORS['purple'], weight='bold')

    legend_texts = ["Motor resonant frequency (ω0): %.1fHz" % (motor_fr), 
                    "Motor damping ratio (ζ): %.3f" % (motor_zeta)]
    for i, text in enumerate(legend_texts):
        ax.text(0.90 + i*0.05, 0.98, text, transform=ax.transAxes, color=KLIPPAIN_COLORS['red_pink'], fontsize=12,
                 fontweight='bold', verticalalignment='top', rotation='vertical', zorder=10)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')

    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper left', prop=fontP)

    return

def plot_vibration_spectrogram_polar(ax, angles, speeds, spectrogram_data):
    angles_radians = np.radians(angles)

    # Assuming speeds defines the radial distance from the center, we need to create a meshgrid
    # for both angles and speeds to map the spectrogram data onto a polar plot correctly
    radius, theta = np.meshgrid(speeds, angles_radians)

    ax.set_title("Polar vibrations heatmap", fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold', va='bottom')
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    ax.pcolormesh(theta, radius, spectrogram_data, norm=matplotlib.colors.LogNorm(), cmap='inferno', shading='auto')
    ax.set_thetagrids([theta * 15 for theta in range(360//15)])
    ax.tick_params(axis='y', which='both', colors='white', labelsize='medium')
    ax.set_ylim([0, max(speeds)])

    # Polar plot doesn't follow the gridspec margin, so we adjust it manually here
    pos = ax.get_position()
    new_pos = [pos.x0 - 0.01, pos.y0 - 0.01, pos.width, pos.height]
    ax.set_position(new_pos)

    return

def plot_vibration_spectrogram(ax, angles, speeds, spectrogram_data, peaks):
    ax.set_title("Vibrations heatmap", fontsize=14, color=KLIPPAIN_COLORS['dark_orange'], weight='bold')
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('Angle (deg)')

    ax.imshow(spectrogram_data, norm=matplotlib.colors.LogNorm(), cmap='inferno',
              aspect='auto', extent=[speeds[0], speeds[-1], angles[0], angles[-1]],
              origin='lower', interpolation='antialiased')
    
    # Add peaks lines in the spectrogram to get hint from peaks found in the first graph
    if peaks is not None:
        for idx, peak in enumerate(peaks):
            ax.axvline(speeds[peak], color='cyan', linewidth=0.75)
            ax.annotate(f"Peak {idx+1}", (speeds[peak], angles[-1]*0.9),
                        textcoords="data", color='cyan', rotation=90, fontsize=10,
                        verticalalignment='top', horizontalalignment='right')

    return


######################################################################
# Startup and main routines
######################################################################

def extract_angle_and_speed(logname):
    try:
        match = re.search(r'an(\d+)_\d+sp(\d+)_\d+', os.path.basename(logname))
        if match:
            angle = match.group(1)
            speed = match.group(2)
    except AttributeError:
        raise ValueError(f"File {logname} does not match expected format.")
    return float(angle), float(speed)


def dir_vibrations_profile(lognames, klipperdir="~/klipper", kinematics="cartesian", accel=None, max_freq=1000.):
    set_locale()
    global shaper_calibrate
    shaper_calibrate = setup_klipper_import(klipperdir)

    if kinematics == "cartesian": main_angles = [0, 90]
    elif kinematics == "corexy": main_angles = [45, 135]
    else:
        raise ValueError("Only Cartesian and CoreXY kinematics are supported by this tool at the moment!")

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
            raise ValueError("Measurements not taken at the correct angles for the specified kinematics!")

    # Precompute the variables used in plot functions
    all_angles, all_speeds, spectrogram_data = compute_dir_speed_spectrogram(measured_speeds, psds_sum, kinematics, main_angles)
    all_angles_energy = compute_angle_powers(spectrogram_data)
    sp_min_energy, sp_max_energy, sp_avg_energy, sp_energy_variance = compute_speed_powers(spectrogram_data)
    motor_profiles, global_motor_profile = compute_motor_profiles(target_freqs, psds, all_angles_energy, main_angles)

    symmetry_factor = compute_symmetry_analysis(all_angles, all_angles_energy)
    print_with_c_locale(f"Machine estimated vibration symmetry: {symmetry_factor:.1f}%")

    # Analyze low variance ranges of vibration energy across all angles for each speed to identify clean speeds
    # and highlight them. Also find the peaks to identify speeds to avoid due to high resonances
    num_peaks, vibration_peaks, peaks_speeds = detect_peaks(
        sp_energy_variance, all_speeds,
        PEAKS_DETECTION_THRESHOLD * sp_energy_variance.max(),
        PEAKS_RELATIVE_HEIGHT_THRESHOLD, 10, 10
        )
    formated_peaks_speeds = ["{:.1f}".format(pspeed) for pspeed in peaks_speeds]
    print_with_c_locale("Vibrations peaks detected: %d @ %s mm/s (avoid setting a speed near these values in your slicer print profile)" % (num_peaks, ", ".join(map(str, formated_peaks_speeds))))
    
    good_speeds = identify_low_energy_zones(sp_energy_variance, SPEEDS_VALLEY_DETECTION_THRESHOLD)
    if good_speeds is not None:
        print_with_c_locale(f'Lowest vibrations speeds ({len(good_speeds)} ranges sorted from best to worse):')
        for idx, (start, end, energy) in enumerate(good_speeds):
            print_with_c_locale(f'{idx+1}: {all_speeds[start]:.1f} to {all_speeds[end]:.1f} mm/s')

    # Angle low energy valleys identification (good angles ranges) and print them to the console
    good_angles = identify_low_energy_zones(all_angles_energy, ANGLES_VALLEY_DETECTION_THRESHOLD)
    if good_angles is not None:
        print_with_c_locale(f'Lowest vibrations angles ({len(good_angles)} ranges sorted from best to worse):')
        for idx, (start, end, energy) in enumerate(good_angles):
            print_with_c_locale(f'{idx+1}: {all_angles[start]:.1f}° to {all_angles[end]:.1f}° (mean vibrations energy: {energy:.2f}% of max)')

    # Create graph layout
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, gridspec_kw={
            'height_ratios':[1, 1],
            'width_ratios':[4, 8, 4],
            'bottom':0.050,
            'top':0.890,
            'left':0.040,
            'right':0.985,
            'hspace':0.166,
            'wspace':0.138
            })

    # Transform ax3 and ax4 to polar plots
    ax3.remove()
    ax3 = fig.add_subplot(2, 3, 3, projection='polar')
    ax4.remove()
    ax4 = fig.add_subplot(2, 3, 4, projection='polar')

    # Set the global .png figure size
    fig.set_size_inches(19, 11.6)

    # Add title
    title_line1 = "MACHINE VIBRATIONS ANALYSIS TOOL"
    fig.text(0.060, 0.965, title_line1, ha='left', va='bottom', fontsize=20, color=KLIPPAIN_COLORS['purple'], weight='bold')
    try:
        filename_parts = (lognames[0].split('/')[-1]).split('_')
        dt = datetime.strptime(f"{filename_parts[1]} {filename_parts[2].split('-')[0]}", "%Y%m%d %H%M%S")
        title_line2 = dt.strftime('%x %X')
        if accel is not None:
            title_line2 += ' at ' + str(accel) + ' mm/s²'
    except:
        print_with_c_locale("Warning: CSV filename look to be different than expected (%s)" % (lognames[0]))
        title_line2 = lognames[0].split('/')[-1]
    fig.text(0.060, 0.957, title_line2, ha='left', va='top', fontsize=16, color=KLIPPAIN_COLORS['dark_purple'])

    # Plot the graphs
    plot_angle_profile_polar(ax3, all_angles, all_angles_energy, good_angles, symmetry_factor)
    plot_vibration_spectrogram_polar(ax4, all_angles, all_speeds, spectrogram_data)

    plot_motor_profiles(ax1, target_freqs, main_angles, motor_profiles, global_motor_profile)
    plot_angle_profile(ax6, all_angles, all_angles_energy, good_angles)
    plot_speed_profile(ax2, all_speeds, sp_min_energy, sp_max_energy, sp_avg_energy, sp_energy_variance, num_peaks, vibration_peaks, good_speeds)

    plot_vibration_spectrogram(ax5, all_angles, all_speeds, spectrogram_data, vibration_peaks)

    # Adding a small Klippain logo to the top left corner of the figure
    ax_logo = fig.add_axes([0.001, 0.924, 0.075, 0.075], anchor='NW')
    ax_logo.imshow(plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'klippain.png')))
    ax_logo.axis('off')

    # Adding Shake&Tune version in the top right corner
    st_version = get_git_version()
    if st_version is not None:
        fig.text(0.995, 0.985, st_version, ha='right', va='bottom', fontsize=8, color=KLIPPAIN_COLORS['purple'])

    return fig


def main():
    # Parse command-line arguments
    usage = "%prog [options] <raw logs>"
    opts = optparse.OptionParser(usage)
    opts.add_option("-o", "--output", type="string", dest="output",
                    default=None, help="filename of output graph")
    opts.add_option("-c", "--accel", type="int", dest="accel",
                    default=None, help="accel value to be printed on the graph")
    opts.add_option("-f", "--max_freq", type="float", default=1000.,
                    help="maximum frequency to graph")
    opts.add_option("-k", "--klipper_dir", type="string", dest="klipperdir",
                    default="~/klipper", help="main klipper directory")
    opts.add_option("-m", "--kinematics", type="string", dest="kinematics",
                    default="cartesian", help="machine kinematics configuration")
    options, args = opts.parse_args()
    if len(args) < 1:
        opts.error("No CSV file(s) to analyse")
    if options.output is None:
        opts.error("You must specify an output file.png to use the script (option -o)")
    if options.kinematics not in ["cartesian", "corexy"]:
        opts.error("Only Cartesian and CoreXY kinematics are supported by this tool at the moment!")

    fig = dir_vibrations_profile(args, options.klipperdir, options.kinematics, options.accel, options.max_freq)
    fig.savefig(options.output, dpi=150)


if __name__ == '__main__':
    main()
