#!/usr/bin/env python3

# Common functions for the Shake&Tune package
# Written by Frix_x#0161 #

import math
import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from git import GitCommandError, Repo
from scipy.signal import spectrogram


def parse_log(logname):
    with open(logname) as f:
        for header in f:
            if not header.startswith('#'):
                break
        if not header.startswith('freq,psd_x,psd_y,psd_z,psd_xyz'):
            # Raw accelerometer data
            return np.loadtxt(logname, comments='#', delimiter=',')
    # Power spectral density data or shaper calibration data
    raise ValueError(
        'File %s does not contain raw accelerometer data and therefore '
        'is not supported by Shake&Tune. Please use the official Klipper '
        'script to process it instead.' % (logname,)
    )


def setup_klipper_import(kdir):
    kdir = os.path.expanduser(kdir)
    sys.path.append(os.path.join(kdir, 'klippy'))
    return import_module('.shaper_calibrate', 'extras')


# This is used to print the current S&T version on top of the png graph file
def get_git_version():
    try:
        # Get the absolute path of the script, resolving any symlinks
        # Then get 2 times to parent dir to be at the git root folder
        script_path = Path(__file__).resolve()
        repo_path = script_path.parents[1]
        repo = Repo(repo_path)

        try:
            version = repo.git.describe('--tags')
        except GitCommandError:
            # If no tag is found, use the simplified commit SHA instead
            version = repo.head.commit.hexsha[:7]
        return version

    except Exception:
        return None


# This is Klipper's spectrogram generation function adapted to use Scipy
def compute_spectrogram(data):
    N = data.shape[0]
    Fs = N / (data[-1, 0] - data[0, 0])
    # Round up to a power of 2 for faster FFT
    M = 1 << int(0.5 * Fs - 1).bit_length()
    window = np.kaiser(M, 6.0)

    def _specgram(x):
        return spectrogram(
            x, fs=Fs, window=window, nperseg=M, noverlap=M // 2, detrend='constant', scaling='density', mode='psd'
        )

    d = {'x': data[:, 1], 'y': data[:, 2], 'z': data[:, 3]}
    f, t, pdata = _specgram(d['x'])
    for axis in 'yz':
        pdata += _specgram(d[axis])[2]
    return pdata, t, f


# Compute natural resonant frequency and damping ratio by using the half power bandwidth method with interpolated frequencies
def compute_mechanical_parameters(psd, freqs, min_freq=None):
    max_under_min_freq = False

    if min_freq is not None:
        min_freq_index = np.searchsorted(freqs, min_freq, side='left')
        if min_freq_index >= len(freqs):
            return None, None, None, max_under_min_freq
        if np.argmax(psd) < min_freq_index:
            max_under_min_freq = True
    else:
        min_freq_index = 0

    # Consider only the part of the signal above min_freq
    psd_above_min_freq = psd[min_freq_index:]
    if len(psd_above_min_freq) == 0:
        return None, None, None, max_under_min_freq

    max_power_index_above_min_freq = np.argmax(psd_above_min_freq)
    max_power_index = max_power_index_above_min_freq + min_freq_index
    fr = freqs[max_power_index]
    max_power = psd[max_power_index]

    half_power = max_power / math.sqrt(2)
    indices_below = np.where(psd[:max_power_index] <= half_power)[0]
    indices_above = np.where(psd[max_power_index:] <= half_power)[0]

    # If we are not able to find points around the half power, we can't compute the damping ratio and return None instead
    if len(indices_below) == 0 or len(indices_above) == 0:
        return fr, None, max_power_index, max_under_min_freq

    idx_below = indices_below[-1]
    idx_above = indices_above[0] + max_power_index
    freq_below_half_power = freqs[idx_below] + (half_power - psd[idx_below]) * (
        freqs[idx_below + 1] - freqs[idx_below]
    ) / (psd[idx_below + 1] - psd[idx_below])
    freq_above_half_power = freqs[idx_above - 1] + (half_power - psd[idx_above - 1]) * (
        freqs[idx_above] - freqs[idx_above - 1]
    ) / (psd[idx_above] - psd[idx_above - 1])

    bandwidth = freq_above_half_power - freq_below_half_power
    bw1 = math.pow(bandwidth / fr, 2)
    bw2 = math.pow(bandwidth / fr, 4)

    try:
        zeta = math.sqrt(0.5 - math.sqrt(1 / (4 + 4 * bw1 - bw2)))
    except ValueError:
        # If a math problem arise such as a negative sqrt term, we also return None instead for damping ratio
        return fr, None, max_power_index, max_under_min_freq

    return fr, zeta, max_power_index, max_under_min_freq


# This find all the peaks in a curve by looking at when the derivative term goes from positive to negative
# Then only the peaks found above a threshold are kept to avoid capturing peaks in the low amplitude noise of a signal
def detect_peaks(data, indices, detection_threshold, relative_height_threshold=None, window_size=5, vicinity=3):
    # Smooth the curve using a moving average to avoid catching peaks everywhere in noisy signals
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, kernel, mode='valid')
    mean_pad = [np.mean(data[:window_size])] * (window_size // 2)
    smoothed_data = np.concatenate((mean_pad, smoothed_data))

    # Find peaks on the smoothed curve
    smoothed_peaks = (
        np.where((smoothed_data[:-2] < smoothed_data[1:-1]) & (smoothed_data[1:-1] > smoothed_data[2:]))[0] + 1
    )
    smoothed_peaks = smoothed_peaks[smoothed_data[smoothed_peaks] > detection_threshold]

    # Additional validation for peaks based on relative height
    valid_peaks = smoothed_peaks
    if relative_height_threshold is not None:
        valid_peaks = []
        for peak in smoothed_peaks:
            peak_height = smoothed_data[peak] - np.min(
                smoothed_data[max(0, peak - vicinity) : min(len(smoothed_data), peak + vicinity + 1)]
            )
            if peak_height > relative_height_threshold * smoothed_data[peak]:
                valid_peaks.append(peak)

    # Refine peak positions on the original curve
    refined_peaks = []
    for peak in valid_peaks:
        local_max = peak + np.argmax(data[max(0, peak - vicinity) : min(len(data), peak + vicinity + 1)]) - vicinity
        refined_peaks.append(local_max)

    num_peaks = len(refined_peaks)

    return num_peaks, np.array(refined_peaks), indices[refined_peaks]


# The goal is to find zone outside of peaks (flat low energy zones) in a signal
def identify_low_energy_zones(power_total, detection_threshold=0.1):
    valleys = []

    # Calculate the a "mean + 1/4" and standard deviation of the entire power_total
    mean_energy = np.mean(power_total) + (np.max(power_total) - np.min(power_total)) / 4
    std_energy = np.std(power_total)

    # Define a threshold value as "mean + 1/4" minus a certain number of standard deviations
    threshold_value = mean_energy - detection_threshold * std_energy

    # Find valleys in power_total based on the threshold
    in_valley = False
    start_idx = 0
    for i, value in enumerate(power_total):
        if not in_valley and value < threshold_value:
            in_valley = True
            start_idx = i
        elif in_valley and value >= threshold_value:
            in_valley = False
            valleys.append((start_idx, i))

    # If the last point is still in a valley, close the valley
    if in_valley:
        valleys.append((start_idx, len(power_total) - 1))

    max_signal = np.max(power_total)

    # Calculate mean energy for each valley as a percentage of the maximum of the signal
    valley_means_percentage = []
    for start, end in valleys:
        if not np.isnan(np.mean(power_total[start:end])):
            valley_means_percentage.append((start, end, (np.mean(power_total[start:end]) / max_signal) * 100))

    # Sort valleys based on mean percentage values
    sorted_valleys = sorted(valley_means_percentage, key=lambda x: x[2])

    return sorted_valleys


# Calculate or estimate a "similarity" factor between two PSD curves and scale it to a percentage. This is
# used here to quantify how close the two belts path behavior and responses are close together.
def compute_curve_similarity_factor(x1, y1, x2, y2, sim_sigmoid_k=0.6):
    # Interpolate PSDs to match the same frequency bins and do a cross-correlation
    y2_interp = np.interp(x1, x2, y2)
    cross_corr = np.correlate(y1, y2_interp, mode='full')

    # Find the peak of the cross-correlation and compute a similarity normalized by the energy of the signals
    peak_value = np.max(cross_corr)
    similarity = peak_value / (np.sqrt(np.sum(y1**2) * np.sum(y2_interp**2)))

    # Apply sigmoid scaling to get better numbers and get a final percentage value
    scaled_similarity = sigmoid_scale(-np.log(1 - similarity), sim_sigmoid_k)

    return scaled_similarity


# Simple helper to compute a sigmoid scalling (from 0 to 100%)
def sigmoid_scale(x, k=1):
    return 1 / (1 + np.exp(-k * x)) * 100
