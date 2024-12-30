# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: static_graph_creator.py
# Description: Implements a static frequency profile measurement script for Shake&Tune to diagnose mechanical
#              issues, including computation and graphing functions for 3D printer vibration analysis.


from typing import List, Optional

import numpy as np

from ..helpers.accelerometer import Measurement, MeasurementsManager
from ..helpers.common_func import compute_spectrogram
from ..helpers.console_output import ConsoleOutput
from ..shaketune_config import ShakeTuneConfig
from .graph_creator import GraphCreator


@GraphCreator.register('static frequency')
class StaticGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config)
        self._freq: Optional[float] = None
        self._duration: Optional[float] = None
        self._accel_per_hz: Optional[float] = None

    def configure(self, freq: float = None, duration: float = None, accel_per_hz: Optional[float] = None) -> None:
        self._freq = freq
        self._duration = duration
        self._accel_per_hz = accel_per_hz

    def create_graph(self, measurements_manager: MeasurementsManager) -> None:
        computer = StaticGraphComputation(
            measurements=measurements_manager.get_measurements(),
            freq=self._freq,
            duration=self._duration,
            max_freq=self._config.max_freq,
            accel_per_hz=self._accel_per_hz,
            st_version=self._version,
        )
        computation = computer.compute()
        fig = self._plotter.plot_static_frequency_graph(computation)
        try:
            axis_label = (measurements_manager.get_measurements())[0]['name'].split('_')[1]
        except IndexError:
            axis_label = None
        self._save_figure(fig, measurements_manager, axis_label)


class StaticGraphComputation:
    def __init__(
        self,
        measurements: List[Measurement],
        freq: float,
        duration: float,
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

    def compute(self):
        if len(self.measurements) == 0:
            raise ValueError('No valid data found in the provided measurements!')
        if len(self.measurements) > 1:
            ConsoleOutput.print('Warning: incorrect number of measurements detected. Only the first one will be used!')

        datas = [np.array(m['samples']) for m in self.measurements if m['samples'] is not None]

        pdata, bins, t = compute_spectrogram(datas[0])
        del datas

        return {
            'freq': self.freq,
            'duration': self.duration,
            'accel_per_hz': self.accel_per_hz,
            'st_version': self.st_version,
            'measurements': self.measurements,
            't': t,
            'bins': bins,
            'pdata': pdata,
            'max_freq': self.max_freq,
        }
