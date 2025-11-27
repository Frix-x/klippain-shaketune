# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: belts_computation.py
# Description: Computation implementation for belts comparison analysis

from typing import Any, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr

from ...helpers.accelerometer import Measurement
from ...helpers.common_func import detect_peaks
from ...helpers.console_output import ConsoleOutput
from .. import get_shaper_calibrate_module
from ..base_models import GraphMetadata
from ..computation_results import BeltsResult, SignalData

PEAKS_DETECTION_THRESHOLD = 0.1  # Threshold to detect peaks in the PSD signal (10% of max)
DC_MAX_PEAKS = 2  # Maximum ideal number of peaks
DC_MAX_UNPAIRED_PEAKS_ALLOWED = 0  # No unpaired peaks are tolerated


class PeakPairingResult(NamedTuple):
    """Result from peak pairing algorithm"""

    paired_peaks: List[Tuple[Tuple[int, float, float], Tuple[int, float, float]]]
    unpaired_peaks1: List[int]
    unpaired_peaks2: List[int]


class BeltsComputation:
    """Computation for belts comparison analysis"""

    def __init__(
        self,
        measurements: List[Measurement],
        kinematics: Optional[str],
        max_freq: float,
        test_params: Optional[Any],
        max_scale: Optional[int],
        st_version: str,
    ):
        self.measurements = measurements
        self.kinematics = kinematics
        self.max_freq = max_freq
        self.test_params = test_params
        self.max_scale = max_scale
        self.st_version = st_version

    def compute(self) -> BeltsResult:
        """Perform belts comparison computation"""
        if len(self.measurements) != 2:
            raise ValueError(
                f'This tool needs 2 measurements to work with (one for each belt)! Currently, it has {len(self.measurements)} '
                f'measurements named {[meas.get("name", "unknown") for meas in self.measurements]}'
            )

        datas = [np.array(m['samples']) for m in self.measurements if m['samples'] is not None]

        # Get belt names for labels
        belt_info = {'A': ' (axis 1,-1)', 'B': ' (axis 1, 1)'}
        signal1_belt = self.measurements[0]['name'].split('_')[1]
        signal2_belt = self.measurements[1]['name'].split('_')[1]
        signal1_belt += belt_info.get(signal1_belt, '')
        signal2_belt += belt_info.get(signal2_belt, '')

        # Compute calibration data
        common_freqs = np.linspace(0, self.max_freq, 500)
        signal1 = self._compute_signal_data(datas[0], common_freqs, self.max_freq)
        signal2 = self._compute_signal_data(datas[1], common_freqs, self.max_freq)
        del datas

        # Pair peaks
        pairing_result = self._pair_peaks(
            signal1.peaks, signal1.freqs, signal1.psd, signal2.peaks, signal2.freqs, signal2.psd
        )
        signal1 = SignalData(
            freqs=signal1.freqs,
            psd=signal1.psd,
            peaks=signal1.peaks,
            paired_peaks=pairing_result.paired_peaks,
            unpaired_peaks=pairing_result.unpaired_peaks1,
        )
        signal2 = SignalData(
            freqs=signal2.freqs,
            psd=signal2.psd,
            peaks=signal2.peaks,
            paired_peaks=pairing_result.paired_peaks,
            unpaired_peaks=pairing_result.unpaired_peaks2,
        )

        # Compute similarity factor and MHI if needed (for symmetric kinematics)
        similarity_factor = None
        mhi = None
        if self.kinematics in {'limited_corexy', 'corexy', 'limited_corexz', 'corexz'}:
            correlation, _ = pearsonr(signal1.psd, signal2.psd)
            similarity_factor = correlation * 100
            similarity_factor = np.clip(similarity_factor, 0, 100)
            ConsoleOutput.print(f'Belts estimated similarity: {similarity_factor:.1f}%')

            mhi = self._compute_mhi(similarity_factor, signal1, signal2)
            ConsoleOutput.print(f'Mechanical health: {mhi}')

        # Create metadata
        metadata = GraphMetadata(
            title='RELATIVE BELTS CALIBRATION TOOL',
            version=self.st_version,
            additional_info={
                'kinematics': self.kinematics,
                'test_params': self.test_params,
            },
        )

        return BeltsResult(
            metadata=metadata,
            measurements=self.measurements,
            signal1=signal1,
            signal2=signal2,
            signal1_belt=signal1_belt,
            signal2_belt=signal2_belt,
            kinematics=self.kinematics,
            test_params=self.test_params,
            max_freq=self.max_freq,
            max_scale=self.max_scale,
            similarity_factor=similarity_factor,
            mhi=mhi,
        )

    def _compute_signal_data(self, data: np.ndarray, common_freqs: np.ndarray, max_freq: float) -> SignalData:
        """Compute signal data from raw measurements"""
        shaper_calibrate, _ = get_shaper_calibrate_module()
        calibration_data = shaper_calibrate.process_accelerometer_data(None, data)

        freqs = calibration_data.freq_bins[calibration_data.freq_bins <= max_freq]
        psd = calibration_data.get_psd('all')[calibration_data.freq_bins <= max_freq]

        # Re-interpolate the PSD signal to a common frequency range
        interp_psd = np.interp(common_freqs, freqs, psd)

        _, peaks, _ = detect_peaks(
            interp_psd,
            common_freqs,
            PEAKS_DETECTION_THRESHOLD * interp_psd.max(),
            window_size=20,
            vicinity=15,
        )

        return SignalData(freqs=common_freqs, psd=interp_psd, peaks=peaks)

    def _pair_peaks(
        self,
        peaks1: np.ndarray,
        freqs1: np.ndarray,
        psd1: np.ndarray,
        peaks2: np.ndarray,
        freqs2: np.ndarray,
        psd2: np.ndarray,
    ) -> PeakPairingResult:
        """Pair peaks between two signals based on frequency proximity"""
        # Compute a dynamic detection threshold
        distances = []
        for p1 in peaks1:
            for p2 in peaks2:
                distances.append(abs(freqs1[p1] - freqs2[p2]))
        distances = np.array(distances)

        if len(distances) > 0:
            median_distance = np.median(distances)
            iqr = np.percentile(distances, 75) - np.percentile(distances, 25)
            threshold = median_distance + 1.5 * iqr
            threshold = min(threshold, 10)
        else:
            threshold = 10

        # Pair the peaks using the dynamic threshold
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

        return PeakPairingResult(
            paired_peaks=paired_peaks, unpaired_peaks1=unpaired_peaks1, unpaired_peaks2=unpaired_peaks2
        )

    def _compute_mhi(self, similarity_factor: float, signal1: SignalData, signal2: SignalData) -> str:
        """Compute Mechanical Health Indicator"""
        num_unpaired_peaks = len(signal1.unpaired_peaks) + len(signal2.unpaired_peaks)
        num_paired_peaks = len(signal1.paired_peaks)

        # Combine unpaired peaks from both signals
        combined_unpaired_peaks = [(peak, signal1) for peak in signal1.unpaired_peaks] + [
            (peak, signal2) for peak in signal2.unpaired_peaks
        ]
        psd_highest_max = max(signal1.psd.max(), signal2.psd.max())

        # Start with the similarity factor directly scaled to a percentage
        mhi = similarity_factor

        # Bonus for ideal number of total peaks (1 or 2)
        if num_paired_peaks >= DC_MAX_PEAKS:
            mhi *= DC_MAX_PEAKS / num_paired_peaks  # Reduce MHI if more than ideal number of peaks

        # Penalty from unpaired peaks weighted by their amplitude
        unpaired_peak_penalty = 0
        if num_unpaired_peaks > DC_MAX_UNPAIRED_PEAKS_ALLOWED:
            for peak, signal in combined_unpaired_peaks:
                unpaired_peak_penalty += (signal.psd[peak] / psd_highest_max) * 30
            mhi -= unpaired_peak_penalty

        # Ensure the result lies between 0 and 100
        mhi = np.clip(mhi, 0, 100)

        return self._mhi_lut(mhi)

    def _mhi_lut(self, mhi: float) -> str:
        """Convert MHI value to textual description"""
        ranges = [
            (70, 100, 'Excellent mechanical health'),
            (55, 70, 'Good mechanical health'),
            (45, 55, 'Acceptable mechanical health'),
            (30, 45, 'Potential signs of a mechanical issue'),
            (15, 30, 'Likely a mechanical issue'),
            (0, 15, 'Mechanical issue detected'),
        ]
        mhi = np.clip(mhi, 0, 100)
        return next(
            (message for lower, upper, message in ranges if lower < mhi <= upper),
            'Unknown mechanical health',
        )
