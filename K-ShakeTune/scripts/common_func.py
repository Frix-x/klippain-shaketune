#!/usr/bin/env python3

# Common functions for the Shake&Tune package
# Written by Frix_x#0161 #


import numpy as np
from scipy.signal import spectrogram


# This is Klipper's spectrogram generation function adapted to use Scipy
def compute_spectrogram(data):
    N = data.shape[0]
    Fs = N / (data[-1, 0] - data[0, 0])
    # Round up to a power of 2 for faster FFT
    M = 1 << int(.5 * Fs - 1).bit_length()
    window = np.kaiser(M, 6.)

    def _specgram(x):
        x_detrended = x - np.mean(x)  # Detrending by subtracting the mean value
        return spectrogram(
            x_detrended, fs=Fs, window=window, nperseg=M, noverlap=M//2,
            detrend='constant', scaling='density', mode='psd')

    d = {'x': data[:, 1], 'y': data[:, 2], 'z': data[:, 3]}
    f, t, pdata = _specgram(d['x'])
    for axis in 'yz':
        pdata += _specgram(d[axis])[2]
    return pdata, t, f


# This find all the peaks in a curve by looking at when the derivative term goes from positive to negative
# Then only the peaks found above a threshold are kept to avoid capturing peaks in the low amplitude noise of a signal
def detect_peaks(data, indices, detection_threshold, relative_height_threshold=None, window_size=5, vicinity=3):
    # Smooth the curve using a moving average to avoid catching peaks everywhere in noisy signals
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, kernel, mode='valid')
    mean_pad = [np.mean(data[:window_size])] * (window_size // 2)
    smoothed_data = np.concatenate((mean_pad, smoothed_data))
    
    # Find peaks on the smoothed curve
    smoothed_peaks = np.where((smoothed_data[:-2] < smoothed_data[1:-1]) & (smoothed_data[1:-1] > smoothed_data[2:]))[0] + 1
    smoothed_peaks = smoothed_peaks[smoothed_data[smoothed_peaks] > detection_threshold]
    
    # Additional validation for peaks based on relative height
    valid_peaks = smoothed_peaks
    if relative_height_threshold is not None:
        valid_peaks = []
        for peak in smoothed_peaks:
            peak_height = smoothed_data[peak] - np.min(smoothed_data[max(0, peak-vicinity):min(len(smoothed_data), peak+vicinity+1)])
            if peak_height > relative_height_threshold * smoothed_data[peak]:
                valid_peaks.append(peak)

    # Refine peak positions on the original curve
    refined_peaks = []
    for peak in valid_peaks:
        local_max = peak + np.argmax(data[max(0, peak-vicinity):min(len(data), peak+vicinity+1)]) - vicinity
        refined_peaks.append(local_max)
    
    num_peaks = len(refined_peaks)
    
    return num_peaks, np.array(refined_peaks), indices[refined_peaks]
