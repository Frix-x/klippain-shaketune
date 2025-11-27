# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: shaper_computation.py
# Description: Computation implementation for input shaper calibration

from typing import Any, List, Optional

import numpy as np

from ...helpers.accelerometer import Measurement
from ...helpers.common_func import compute_mechanical_parameters, compute_spectrogram, detect_peaks
from ...helpers.console_output import ConsoleOutput
from .. import get_shaper_calibrate_module
from ..base_models import GraphMetadata
from ..computation_results import ShaperResult

PEAKS_DETECTION_THRESHOLD = 0.05
PEAKS_EFFECT_THRESHOLD = 0.12
MAX_VIBRATIONS = 5.0
MIN_SMOOTHING = 0.001


class ShaperComputation:
    """Computation for input shaper calibration"""

    def __init__(
        self,
        measurements: List[Measurement],
        max_smoothing: Optional[float],
        scv: float,
        max_freq: float,
        test_params: Optional[Any],
        max_scale: Optional[float],
        st_version: str,
    ):
        self.measurements = measurements
        self.max_smoothing = max_smoothing
        self.scv = scv
        self.max_freq = max_freq
        self.test_params = test_params
        self.max_scale = max_scale
        self.st_version = st_version

    def compute(self) -> ShaperResult:
        """Perform input shaper computation"""
        if len(self.measurements) == 0:
            raise ValueError('No valid data found in the provided measurements!')
        if len(self.measurements) > 1:
            ConsoleOutput.print('Warning: incorrect number of measurements detected. Only the first one will be used!')

        datas = [np.array(m['samples']) for m in self.measurements if m['samples'] is not None]

        # Compute shapers, PSD outputs and spectrogram
        (
            k_shaper_choice,
            k_shapers,
            shapers_tradeoff_data,
            calibration_data,
            fr,
            zeta,
            compat,
        ) = self._calibrate_shaper(datas[0], self.max_smoothing, self.scv, self.max_freq)
        pdata, bins, t = compute_spectrogram(datas[0])
        del datas

        # Select only the relevant part of the PSD data
        freqs = calibration_data.freq_bins
        calibration_data.psd_sum = calibration_data.psd_sum[freqs <= self.max_freq]
        calibration_data.psd_x = calibration_data.psd_x[freqs <= self.max_freq]
        calibration_data.psd_y = calibration_data.psd_y[freqs <= self.max_freq]
        calibration_data.psd_z = calibration_data.psd_z[freqs <= self.max_freq]
        calibration_data.freqs = freqs[freqs <= self.max_freq]

        # Peak detection algorithm
        peaks_threshold = [
            PEAKS_DETECTION_THRESHOLD * calibration_data.psd_sum.max(),
            PEAKS_EFFECT_THRESHOLD * calibration_data.psd_sum.max(),
        ]
        num_peaks, peaks, peaks_freqs = detect_peaks(
            calibration_data.psd_sum, calibration_data.freqs, peaks_threshold[0]
        )

        # Print the peaks info in the console
        peak_freqs_formated = ['{:.1f}'.format(f) for f in peaks_freqs]
        num_peaks_above_effect_threshold = np.sum(calibration_data.psd_sum[peaks] > peaks_threshold[1])
        ConsoleOutput.print(
            f'Peaks detected on the graph: {num_peaks} @ {", ".join(map(str, peak_freqs_formated))} Hz ({num_peaks_above_effect_threshold} above effect threshold)'
        )

        # Consolidate shaper data for plotting the table summary
        # and data for the shaper recommendation (performance vs low vibration)
        shaper_table_data = {
            'shapers': [],
            'recommendations': [],
            'damping_ratio': zeta,
        }

        perf_shaper_choice = None
        perf_shaper_freq = None
        perf_shaper_accel = 0
        max_smoothing_computed = 0
        for shaper in k_shapers:
            shaper_info = {
                'type': shaper.name.upper(),
                'frequency': shaper.freq,
                'vibrations': shaper.vibrs,
                'smoothing': shaper.smoothing,
                'max_accel': shaper.max_accel,
                'vals': shaper.vals,
            }
            shaper_table_data['shapers'].append(shaper_info)
            max_smoothing_computed = max(max_smoothing_computed, shaper.smoothing)

            # Get the Klipper recommended shaper (usually it's a good low vibration compromise)
            if shaper.name == k_shaper_choice:
                klipper_shaper_freq = shaper.freq
                klipper_shaper_accel = shaper.max_accel

            # Find the shaper with the highest accel but with vibrs under MAX_VIBRATIONS as it's
            # a good performance compromise when injecting the SCV and damping ratio in the computation
            if perf_shaper_accel < shaper.max_accel and shaper.vibrs * 100 < MAX_VIBRATIONS:
                perf_shaper_choice = shaper.name
                perf_shaper_accel = shaper.max_accel
                perf_shaper_freq = shaper.freq

        # Recommendations are put in the console: one is Klipper's original suggestion that is usually good for low vibrations
        # and the other one is the custom "performance" recommendation that looks for a suitable shaper that doesn't have excessive
        # vibrations level but have higher accelerations. If both recommendations are the same shaper, or if no suitable "performance"
        # shaper is found, then only a single line as the "best shaper" recommendation is printed
        ConsoleOutput.print('Recommended filters:')
        if (
            perf_shaper_choice is not None
            and perf_shaper_choice != k_shaper_choice
            and perf_shaper_accel >= klipper_shaper_accel
        ):
            perf_shaper_string = f'    -> For performance: {perf_shaper_choice.upper()} @ {perf_shaper_freq:.1f} Hz'
            lowvibr_shaper_string = (
                f'    -> For low vibrations: {k_shaper_choice.upper()} @ {klipper_shaper_freq:.1f} Hz'
            )
            shaper_table_data['recommendations'].append(perf_shaper_string)
            shaper_table_data['recommendations'].append(lowvibr_shaper_string)
            shaper_choices = [k_shaper_choice.upper(), perf_shaper_choice.upper()]
            ConsoleOutput.print(f'{perf_shaper_string} (with a damping ratio of {zeta:.3f})')
            ConsoleOutput.print(f'{lowvibr_shaper_string} (with a damping ratio of {zeta:.3f})')
        else:
            shaper_string = f'    -> Best shaper: {k_shaper_choice.upper()} @ {klipper_shaper_freq:.1f} Hz'
            shaper_table_data['recommendations'].append(shaper_string)
            shaper_choices = [k_shaper_choice.upper()]
            ConsoleOutput.print(f'{shaper_string} (with a damping ratio of {zeta:.3f})')

        # Create metadata
        metadata = GraphMetadata(
            title='INPUT SHAPER CALIBRATION TOOL',
            version=self.st_version,
            additional_info={
                'test_params': self.test_params,
                'scv': self.scv,
                'max_smoothing': self.max_smoothing,
            },
        )

        return ShaperResult(
            metadata=metadata,
            measurements=self.measurements,
            calibration_data=calibration_data,
            shapers=k_shapers,
            shaper_table_data=shaper_table_data,
            shaper_choices=shaper_choices,
            peaks=peaks,
            peaks_freqs=peaks_freqs,
            peaks_threshold=peaks_threshold,
            fr=fr,
            zeta=zeta,
            t=t,
            bins=bins,
            pdata=pdata,
            test_params=self.test_params,
            max_smoothing=self.max_smoothing,
            scv=self.scv,
            max_freq=self.max_freq,
            max_scale=self.max_scale,
            compat=compat,
            max_smoothing_computed=max_smoothing_computed,
        )

    def _calibrate_shaper(self, datas: np.ndarray, max_smoothing: Optional[float], scv: float, max_freq: float):
        """Find the best shaper parameters using Klipper's official algorithm"""
        shaper_calibrate, shaper_defs = get_shaper_calibrate_module()
        calib_data = shaper_calibrate.process_accelerometer_data(None, datas)
        calib_data.normalize_to_frequencies()

        # We compute the damping ratio using the Klipper's default value if it fails
        fr, zeta, _, _ = compute_mechanical_parameters(calib_data.psd_sum, calib_data.freq_bins)
        zeta = zeta if zeta is not None else 0.1

        # First we find the best shapers using the Klipper's standard algorithms. This will give us Klipper's
        # best shaper choice and the full list of shapers that are set to the current machine response
        compat = False
        try:
            k_shaper_choice, k_shapers = shaper_calibrate.find_best_shaper(
                calib_data,
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
            k_shaper_choice, k_shapers = shaper_calibrate.find_best_shaper(calib_data, max_smoothing, None)

        # Then in a second time, we run again the same computation but with a super low smoothing value to
        # get the maximum accelerations values for each algorithms.
        if compat:
            _, k_shapers_max = shaper_calibrate.find_best_shaper(calib_data, MIN_SMOOTHING, None)
        else:
            _, k_shapers_max = shaper_calibrate.find_best_shaper(
                calib_data,
                shapers=None,
                damping_ratio=zeta,
                scv=scv,
                shaper_freqs=None,
                max_smoothing=MIN_SMOOTHING,
                test_damping_ratios=None,
                max_freq=max_freq,
                logger=None,
            )

        return (
            k_shaper_choice.name,
            k_shapers,
            None,  # shapers_tradeoff_data - not implemented in this version
            calib_data,
            fr,
            zeta,
            compat,
        )
