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
from ..shaketune_config import ShakeTuneConfig
from . import get_shaper_calibrate_module
from .graph_creator import GraphCreator

PEAKS_DETECTION_THRESHOLD = 0.05
PEAKS_EFFECT_THRESHOLD = 0.12
SMOOTHING_TESTS = 10  # Number of smoothing values to test (it will significantly increase the computation time)
MAX_VIBRATIONS = 5.0


@GraphCreator.register('input shaper')
class ShaperGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config)
        self._max_smoothing: Optional[float] = None
        self._scv: Optional[float] = None
        self._accel_per_hz: Optional[float] = None

    def configure(
        self, scv: float, max_smoothing: Optional[float] = None, accel_per_hz: Optional[float] = None
    ) -> None:
        self._scv = scv
        self._max_smoothing = max_smoothing
        self._accel_per_hz = accel_per_hz

    def create_graph(self, measurements_manager: MeasurementsManager) -> None:
        computer = ShaperGraphComputation(
            measurements=measurements_manager.get_measurements(),
            max_smoothing=self._max_smoothing,
            scv=self._scv,
            accel_per_hz=self._accel_per_hz,
            max_freq=self._config.max_freq,
            st_version=self._version,
        )
        computation = computer.compute()
        fig = self._plotter.plot_input_shaper_graph(computation)
        try:
            axis_label = (measurements_manager.get_measurements())[0]['name'].split('_')[1]
        except IndexError:
            axis_label = None
        self._save_figure(fig, measurements_manager, axis_label)

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
        accel_per_hz: Optional[float],
        scv: float,
        max_smoothing: Optional[float],
        max_freq: float,
        st_version: str,
    ):
        self.measurements = measurements
        self.accel_per_hz = accel_per_hz
        self.scv = scv
        self.max_smoothing = max_smoothing
        self.max_freq = max_freq
        self.st_version = st_version

    def compute(self):
        if len(self.measurements) == 0:
            raise ValueError('No valid data found in the provided measurements!')
        if len(self.measurements) > 1:
            ConsoleOutput.print('Warning: incorrect number of measurements detected. Only the first one will be used!')

        datas = [np.array(m['samples']) for m in self.measurements if m['samples'] is not None]

        # Compute shapers, PSD outputs and spectrogram
        (
            klipper_shaper_choice,
            shapers,
            additional_shapers,
            calibration_data,
            fr,
            zeta,
            max_smoothing_computed,
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
            f"Peaks detected on the graph: {num_peaks} @ {', '.join(map(str, peak_freqs_formated))} Hz ({num_peaks_above_effect_threshold} above effect threshold)"
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
        for shaper in shapers:
            shaper_info = {
                'type': shaper.name.upper(),
                'frequency': shaper.freq,
                'vibrations': shaper.vibrs,
                'smoothing': shaper.smoothing,
                'max_accel': shaper.max_accel,
                'vals': shaper.vals,
            }
            shaper_table_data['shapers'].append(shaper_info)

            # Get the Klipper recommended shaper (usually it's a good low vibration compromise)
            if shaper.name == klipper_shaper_choice:
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
            and perf_shaper_choice != klipper_shaper_choice
            and perf_shaper_accel >= klipper_shaper_accel
        ):
            perf_shaper_string = f'    -> For performance: {perf_shaper_choice.upper()} @ {perf_shaper_freq:.1f} Hz'
            lowvibr_shaper_string = (
                f'    -> For low vibrations: {klipper_shaper_choice.upper()} @ {klipper_shaper_freq:.1f} Hz'
            )
            shaper_table_data['recommendations'].append(perf_shaper_string)
            shaper_table_data['recommendations'].append(lowvibr_shaper_string)
            shaper_choices = [klipper_shaper_choice.upper(), perf_shaper_choice.upper()]
            ConsoleOutput.print(f'{perf_shaper_string} (with a damping ratio of {zeta:.3f})')
            ConsoleOutput.print(f'{lowvibr_shaper_string} (with a damping ratio of {zeta:.3f})')
        else:
            shaper_string = f'    -> Best shaper: {klipper_shaper_choice.upper()} @ {klipper_shaper_freq:.1f} Hz'
            shaper_table_data['recommendations'].append(shaper_string)
            shaper_choices = [klipper_shaper_choice.upper()]
            ConsoleOutput.print(f'{shaper_string} (with a damping ratio of {zeta:.3f})')

        # And finally setup the results to return them
        computation_result = {
            'measurements': self.measurements,
            'compat': compat,
            'max_smoothing_computed': max_smoothing_computed,
            'max_freq': self.max_freq,
            'calibration_data': calibration_data,
            'shapers': shapers,
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
            'additional_shapers': additional_shapers,
            'accel_per_hz': self.accel_per_hz,
            'max_smoothing': self.max_smoothing,
            'scv': self.scv,
            'st_version': self.st_version,
        }

        return computation_result

    # Find the best shaper parameters using Klipper's official algorithm selection with
    # a proper precomputed damping ratio (zeta) and using the configured printer SQV value
    # This function also sweep around the smoothing values to help you find the best compromise
    def _calibrate_shaper(self, datas: List[np.ndarray], max_smoothing: Optional[float], scv: float, max_freq: float):
        helper = get_shaper_calibrate_module().ShaperCalibrate(printer=None)
        calibration_data = helper.process_accelerometer_data(datas)
        calibration_data.normalize_to_frequencies()

        # We compute the damping ratio using the Klipper's default value if it fails
        fr, zeta, _, _ = compute_mechanical_parameters(calibration_data.psd_sum, calibration_data.freq_bins)
        zeta = zeta if zeta is not None else 0.1

        compat = False
        try:
            k_shaper_choice, all_shapers = helper.find_best_shaper(
                calibration_data,
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
            k_shaper_choice, all_shapers = helper.find_best_shaper(calibration_data, max_smoothing, None)

        # If max_smoothing is not None, we run the same computation but without a smoothing value
        # to get the max smoothing values from the filters and create the testing list
        all_shapers_nosmoothing = None
        if max_smoothing is not None:
            if compat:
                _, all_shapers_nosmoothing = helper.find_best_shaper(calibration_data, None, None)
            else:
                _, all_shapers_nosmoothing = helper.find_best_shaper(
                    calibration_data,
                    shapers=None,
                    damping_ratio=zeta,
                    scv=scv,
                    shaper_freqs=None,
                    max_smoothing=None,
                    test_damping_ratios=None,
                    max_freq=max_freq,
                    logger=None,
                )

        # Then we iterate over the all_shaperts_nosmoothing list to get the max of the smoothing values
        max_smoothing = 0.0
        if all_shapers_nosmoothing is not None:
            for shaper in all_shapers_nosmoothing:
                if shaper.smoothing > max_smoothing:
                    max_smoothing = shaper.smoothing
        else:
            for shaper in all_shapers:
                if shaper.smoothing > max_smoothing:
                    max_smoothing = shaper.smoothing

        # Then we create a list of smoothing values to test (no need to test the max smoothing value as it was already tested)
        smoothing_test_list = np.linspace(0.001, max_smoothing, SMOOTHING_TESTS)[:-1]
        additional_all_shapers = {}
        for smoothing in smoothing_test_list:
            if compat:
                _, all_shapers_bis = helper.find_best_shaper(calibration_data, smoothing, None)
            else:
                _, all_shapers_bis = helper.find_best_shaper(
                    calibration_data,
                    shapers=None,
                    damping_ratio=zeta,
                    scv=scv,
                    shaper_freqs=None,
                    max_smoothing=smoothing,
                    test_damping_ratios=None,
                    max_freq=max_freq,
                    logger=None,
                )
            additional_all_shapers[f'sm_{smoothing}'] = all_shapers_bis
        additional_all_shapers['max_smoothing'] = (
            all_shapers_nosmoothing if all_shapers_nosmoothing is not None else all_shapers
        )

        return (
            k_shaper_choice.name,
            all_shapers,
            additional_all_shapers,
            calibration_data,
            fr,
            zeta,
            max_smoothing,
            compat,
        )
