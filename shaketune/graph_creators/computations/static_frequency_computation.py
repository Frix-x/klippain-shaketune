# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: static_frequency_computation.py
# Description: Computation implementation for static frequency analysis

from typing import List, Optional

import numpy as np

from ...helpers.accelerometer import Measurement
from ...helpers.spectrogram import compute_spectrogram
from ...helpers.console_output import ConsoleOutput
from ..base_models import GraphMetadata
from ..computation_results import StaticFrequencyResult


class StaticFrequencyComputation:
    """Computation for static frequency analysis"""

    def __init__(
        self,
        measurements: List[Measurement],
        freq: Optional[float],
        duration: Optional[float],
        max_freq: float,
        accel_per_hz: Optional[float],
        st_version: str,
    ):
        self.measurements = measurements
        self.freq = freq
        self.duration = duration
        self.max_freq = max_freq
        self.accel_per_hz = accel_per_hz
        self.st_version = st_version

    def compute(self) -> StaticFrequencyResult:
        """Perform static frequency computation"""
        if len(self.measurements) == 0:
            raise ValueError('No valid data found in the provided measurements!')

        if len(self.measurements) > 1:
            ConsoleOutput.print('Warning: incorrect number of measurements detected. Only the first one will be used!')

        # Extract data from measurements
        datas = [np.asarray(m['samples']) for m in self.measurements if m['samples'] is not None]

        # Compute spectrogram
        pdata, bins, t = compute_spectrogram(datas[0])
        del datas

        # Create metadata
        metadata = GraphMetadata(
            title='STATIC FREQUENCY HELPER TOOL',
            version=self.st_version,
            additional_info={
                'freq': self.freq,
                'duration': self.duration,
                'accel_per_hz': self.accel_per_hz,
            },
        )

        return StaticFrequencyResult(
            metadata=metadata,
            measurements=self.measurements,
            freq=self.freq,
            duration=self.duration,
            accel_per_hz=self.accel_per_hz,
            t=t,
            bins=bins,
            pdata=pdata,
            max_freq=self.max_freq,
        )
