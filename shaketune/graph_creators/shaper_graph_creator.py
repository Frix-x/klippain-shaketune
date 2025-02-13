# Shake&Tune: 3D printer analysis tools
#
# Derived from the calibrate_shaper.py official Klipper script
# Copyright (C) 2020  Dmitry Butyugin <dmbutyugin@google.com>
# Copyright (C) 2020  Kevin O'Connor <kevin@koconnor.net>
# Copyright (C) 2022 - 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: shaper_graph_creator.py
# Description: Implements the input shaper calibration script for Shake&Tune,
#              including computation and graphing functions for 3D printer vibration analysis.


from typing import List, Optional

import numpy as np

from ..helpers.accelerometer import Measurement, MeasurementsManager
from ..helpers.common_func import (
    compute_mechanical_parameters,
    compute_spectrogram,
    detect_peaks,
)
from ..helpers.console_output import ConsoleOutput
from ..helpers.resonance_test import testParams
from ..shaketune_config import ShakeTuneConfig
from . import get_shaper_calibrate_module
from .graph_creator import GraphCreator

PEAKS_DETECTION_THRESHOLD = 0.05
PEAKS_EFFECT_THRESHOLD = 0.12
MAX_VIBRATIONS = 5.0
MIN_ACCEL = 100.0
MIN_SMOOTHING = 0.001
TARGET_SMOOTHING = 0.12
MIN_FREQ = 5.0
MAX_SHAPER_FREQ = 150.0


@GraphCreator.register('input shaper')
class ShaperGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config)
        self._max_smoothing: Optional[float] = None
        self._scv: Optional[float] = None
        self._test_params: Optional[testParams] = None

    def configure(
        self,
        scv: float,
        max_smoothing: Optional[float] = None,
        test_params: Optional[testParams] = None,
        max_scale: Optional[int] = None,
    ) -> None:
        self._scv = scv
        self._max_smoothing = max_smoothing
        self._test_params = test_params
        self._max_scale = max_scale

    def create_graph(self, measurements_manager: MeasurementsManager) -> None:
        computer = ShaperGraphComputation(
            measurements=measurements_manager.get_measurements(),
            max_smoothing=self._max_smoothing,
            scv=self._scv,
            test_params=self._test_params,
            max_freq=self._config.max_freq,
            max_scale=self._max_scale,
            st_version=self._version,
        )
        computation = computer.compute()
        fig = self._plotter.plot_input_shaper_graph(computation)
        self._save_figure(fig)

    def clean_old_files(self, keep_results: int = 3) -> None:
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)
        if len(files) <= 2 * keep_results:
            return  # No need to delete any files
        for old_png_file in files[2 * keep_results :]:
            stdata_file = old_png_file.with_suffix('.stdata')
            stdata_file.unlink(missing_ok=True)
            old_png_file.unlink()


class ShaperGraphComputation:
    def __init__(
        self,
        measurements: List[Measurement],
        test_params: Optional[testParams],
        scv: float,
        max_smoothing: Optional[float],
        max_freq: float,
        max_scale: Optional[int],
        st_version: str,
    ):
        self.measurements = measurements
        self.test_params = test_params
        self.scv = scv
        self.max_smoothing = max_smoothing
        self.max_freq = max_freq
        self.max_scale = max_scale
        self.st_version = st_version

    def compute(self):
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

        return {
            'measurements': self.measurements,
            'compat': compat,
            'max_smoothing_computed': max_smoothing_computed,
            'max_freq': self.max_freq,
            'calibration_data': calibration_data,
            'shapers': k_shapers,
            'shaper_table_data': shaper_table_data,
            'shaper_choices': shaper_choices,
            'peaks': peaks,
            'peaks_freqs': peaks_freqs,
            'peaks_threshold': peaks_threshold,
            'fr': fr,
            'zeta': zeta,
            't': t,
            'bins': bins,
            'pdata': pdata,
            'shapers_tradeoff_data': shapers_tradeoff_data,
            'test_params': self.test_params,
            'max_smoothing': self.max_smoothing,
            'scv': self.scv,
            'st_version': self.st_version,
            'max_scale': self.max_scale,
        }

    # Find the best shaper parameters using Klipper's official algorithm selection with
    # a proper precomputed damping ratio (zeta) and using the configured printer SQV value
    # This function also sweep around the smoothing values to help you find the best compromise
    def _calibrate_shaper(self, datas: List[np.ndarray], max_smoothing: Optional[float], scv: float, max_freq: float):
        shaper_calibrate, shaper_defs = get_shaper_calibrate_module()
        calib_data = shaper_calibrate.process_accelerometer_data(datas)
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

        # TODO: re-add this in next release
        # --------------------------------------------------------------------------------------------------------------
        # # Finally we create the data structure to plot the additional graphs (tradeoffs for vibrs and smoothing)
        # shapers_tradeoff_data = {}
        # for shaper in k_shapers_max:
        #     shaper_cfg = next(cfg for cfg in shaper_defs.INPUT_SHAPERS if cfg.name == shaper.name)

        #     accels_to_plot = np.linspace(MIN_ACCEL, shaper.max_accel, 100)
        #     smoothing_to_plot = np.linspace(MIN_SMOOTHING, 0.20, 100)
        #     # smoothing_to_plot = np.linspace(MIN_SMOOTHING, shaper.smoothing * 10, 100)

        #     shapers_tradeoff_data[shaper.name] = self._compute_tradeoff_data(
        #         shaper_calibrate,
        #         shaper_cfg,
        #         zeta,
        #         scv,
        #         calib_data,
        #         accels_to_plot,
        #         smoothing_to_plot,
        #     )
        # --------------------------------------------------------------------------------------------------------------

        return (
            k_shaper_choice.name,
            k_shapers,
            None,  # shapers_tradeoff_data,
            calib_data,
            fr,
            zeta,
            compat,
        )

    # # For a given shaper type and machine damping ratio, this compute the remaining vibration and smoothing
    # # the machine will experience for a given range of acceleration values and target smoothing values.
    # def _compute_tradeoff_data(
    #     self, shaper_calibrate, shaper_cfg, zeta, scv, calib_data, accels_to_plot, smoothing_to_plot
    # ):
    #     # Initialize grids to store computed values
    #     vibrations_grid = np.zeros((len(accels_to_plot), len(smoothing_to_plot)))
    #     shaper_freqs_grid = np.zeros_like(vibrations_grid)

    #     # Loop over acceleration and smoothing values
    #     for i, accel in enumerate(accels_to_plot):
    #         for j, smoothing in enumerate(smoothing_to_plot):
    #             # Find the shaper frequency that results in the target smoothing at this acceleration
    #             shaper_freq = self._fit_shaper(
    #                 shaper_calibrate,
    #                 shaper_cfg,
    #                 accel,
    #                 zeta,
    #                 scv,
    #                 target_smoothing=smoothing,
    #                 min_freq=MIN_FREQ,
    #                 max_freq=MAX_SHAPER_FREQ,
    #             )
    #             if shaper_freq is not None:
    #                 A, T = shaper_cfg.init_func(shaper_freq, zeta)
    #                 vibration, _ = shaper_calibrate._estimate_remaining_vibrations(
    #                     (A, T), zeta, calib_data.freq_bins, calib_data.psd_sum
    #                 )
    #                 vibrations_grid[i, j] = vibration * 100.0  # Convert to percentage
    #                 shaper_freqs_grid[i, j] = shaper_freq
    #             else:
    #                 # If no valid frequency is found, store NaN values
    #                 vibrations_grid[i, j] = np.nan
    #                 shaper_freqs_grid[i, j] = np.nan

    #     return {
    #         'accels': accels_to_plot,
    #         'smoothings': smoothing_to_plot,
    #         'vibrations_grid': vibrations_grid,
    #         'shaper_freqs_grid': shaper_freqs_grid,
    #     }

    # # Fit a shaper filter to the accelerometer recording in order to find the shaper frequency that results
    # # (or is close) to the target smoothing at the given acceleration.
    # def _fit_shaper(self, shaper_calibrate, shaper_cfg, accel, zeta, scv, target_smoothing, min_freq, max_freq):
    #     def smoothing_loss(freq):
    #         # Compute shaper parameters
    #         shaper = shaper_cfg.init_func(freq, zeta)
    #         smoothing = shaper_calibrate._get_shaper_smoothing(shaper, accel, scv)
    #         return smoothing - target_smoothing

    #     # Use a root-finding method to solve for freq where smoothing equals target_smoothing
    #     try:
    #         shaper_freq = optimize.brentq(smoothing_loss, min_freq, max_freq)
    #     except ValueError:
    #         shaper_freq = None
    #     return shaper_freq
