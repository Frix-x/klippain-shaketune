#!/usr/bin/env python3

######################################
###### AXE_MAP DETECTION SCRIPT ######
######################################
# Written by Frix_x#0161 #

# Be sure to make this script executable using SSH: type 'chmod +x ./analyze_axesmap.py' when in the folder !

#####################################################################
################ !!! DO NOT EDIT BELOW THIS LINE !!! ################
#####################################################################

import optparse
import numpy as np
import locale
from scipy.signal import butter, filtfilt


NUM_POINTS = 500


# Set the best locale for time and date formating (generation of the titles)
try:
    locale.setlocale(locale.LC_TIME, locale.getdefaultlocale())
except locale.Error:
    locale.setlocale(locale.LC_TIME, 'C')

# Override the built-in print function to avoid problem in Klipper due to locale settings
original_print = print
def print_with_c_locale(*args, **kwargs):
    original_locale = locale.setlocale(locale.LC_ALL, None)
    locale.setlocale(locale.LC_ALL, 'C')
    original_print(*args, **kwargs)
    locale.setlocale(locale.LC_ALL, original_locale)
print = print_with_c_locale


######################################################################
# Computation
######################################################################

def accel_signal_filter(data, cutoff=2, fs=100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    filtered_data -= np.mean(filtered_data)
    return filtered_data

def find_first_spike(data):
    min_index, max_index = np.argmin(data), np.argmax(data)
    return ('-', min_index) if min_index < max_index else ('', max_index)

def get_movement_vector(data, start_idx, end_idx):
    if start_idx < end_idx:
        vector = []
        for i in range(3):
            vector.append(np.mean(data[i][start_idx:end_idx], axis=0))
        return vector
    else:
        return np.zeros(3)

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def compute_errors(filtered_data, spikes_sorted, accel_value, num_points):
    # Get the movement start points in the correct order from the sorted bag of spikes
    movement_starts = [spike[0][1] for spike in spikes_sorted]

    # Theoretical unit vectors for X, Y, Z printer axes
    printer_axes  = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }

    alignment_errors = {}
    sensitivity_errors = {}
    for i, axis in enumerate(['x', 'y', 'z']):
        movement_start = movement_starts[i]
        movement_end = movement_start + num_points
        movement_vector = get_movement_vector(filtered_data, movement_start, movement_end)
        alignment_errors[axis] = angle_between(movement_vector, printer_axes[axis])

        measured_accel_magnitude = np.linalg.norm(movement_vector)
        if accel_value != 0:
            sensitivity_errors[axis] = abs(measured_accel_magnitude - accel_value) / accel_value * 100
        else:
            sensitivity_errors[axis] = None

    return alignment_errors, sensitivity_errors


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


def axesmap_calibration(lognames, accel=None):
    # Parse the raw data and get them ready for analysis
    raw_datas = [parse_log(filename) for filename in lognames]
    if len(raw_datas) > 1:
        raise ValueError("Analysis of multiple CSV files at once is not possible with this script")

    filtered_data = [accel_signal_filter(raw_datas[0][:, i+1]) for i in range(3)]
    spikes = [find_first_spike(filtered_data[i]) for i in range(3)]
    spikes_sorted = sorted([(spikes[0], 'x'), (spikes[1], 'y'), (spikes[2], 'z')], key=lambda x: x[0][1])

    # Using the previous variables to get the axes_map and errors
    axes_map = ','.join([f"{spike[0][0]}{spike[1]}" for spike in spikes_sorted])
    # alignment_error, sensitivity_error = compute_errors(filtered_data, spikes_sorted, accel, NUM_POINTS)

    results = f"Detected axes_map:\n  {axes_map}\n"

    # TODO: work on this function that is currently not giving good results...
    # results += "Accelerometer angle deviation:\n"
    # for axis, angle in alignment_error.items():
    #     angle_degrees = np.degrees(angle) # Convert radians to degrees
    #     results += f"  {axis.upper()} axis: {angle_degrees:.2f} degrees\n"

    # results += "Accelerometer sensitivity error:\n"
    # for axis, error in sensitivity_error.items():
    #     results += f"  {axis.upper()} axis: {error:.2f}%\n"

    return results


def main():
    # Parse command-line arguments
    usage = "%prog [options] <raw logs>"
    opts = optparse.OptionParser(usage)
    opts.add_option("-o", "--output", type="string", dest="output",
                    default=None, help="filename of output graph")
    opts.add_option("-a", "--accel", type="string", dest="accel",
                    default=None, help="acceleration value used to do the movements")
    options, args = opts.parse_args()
    if len(args) < 1:
        opts.error("No CSV file(s) to analyse")
    if options.accel is None:
        opts.error("You must specify the acceleration value used when generating the CSV file (option -a)")
    try:
        accel_value = float(options.accel)
    except ValueError:
        opts.error("Invalid acceleration value. It should be a numeric value.")

    results = axesmap_calibration(args, accel_value)
    print(results)

    if options.output is not None:
        with open(options.output, 'w') as f:
            f.write(results)


if __name__ == '__main__':
    main()
