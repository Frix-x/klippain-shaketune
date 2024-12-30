# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: graph_creator.py
# Description: Abstract base class for creating various types of graphs in Shake&Tune,
#              including methods for moving, preparing, saving, and cleaning up files.
#              This class is inherited by the AxesMapGraphCreator, BeltsGraphCreator,
#              ShaperGraphCreator, VibrationsGraphCreator, StaticGraphCreator


import abc
import os
from datetime import datetime
from typing import Optional

from matplotlib.figure import Figure

from ..helpers.accelerometer import MeasurementsManager
from ..shaketune_config import ShakeTuneConfig
from .plotter import Plotter


class GraphCreator(abc.ABC):
    registry = {}

    @classmethod
    def register(cls, graph_type: str):
        def decorator(subclass):
            cls.registry[graph_type] = subclass
            subclass.graph_type = graph_type
            return subclass

        return decorator

    def __init__(self, config: ShakeTuneConfig):
        self._config = config
        self._graph_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._version = ShakeTuneConfig.get_git_version()
        self._type = self.__class__.graph_type
        self._folder = self._config.get_results_folder(self._type)
        self._plotter = Plotter()
        self._custom_filepath = None

    def _save_figure(
        self, fig: Figure, measurements_manager: MeasurementsManager, axis_label: Optional[str] = None
    ) -> None:
        if os.environ.get('SHAKETUNE_IN_CLI') == '1' and self._custom_filepath is not None:
            fig.savefig(f'{self._custom_filepath}', dpi=self._config.dpi)
        else:
            axis_suffix = f'_{axis_label}' if axis_label else ''
            filename = self._folder / f"{self._type.replace(' ', '')}_{self._graph_date}{axis_suffix}"
            fig.savefig(f'{filename}.png', dpi=self._config.dpi)

        if self._config.keep_raw_data and os.environ.get('SHAKETUNE_IN_CLI') != '1':
            measurements_manager.save_stdata(f'{filename}.stdata')

    def get_type(self) -> str:
        return self._type

    def override_output_target(self, filepath: str) -> None:
        self._custom_filepath = filepath

    @abc.abstractmethod
    def create_graph(self, measurements_manager: MeasurementsManager) -> None:
        pass

    def clean_old_files(self, keep_results: int = 10) -> None:
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)
        if len(files) <= keep_results:
            return  # No need to delete any files
        for old_png_file in files[keep_results:]:
            stdata_file = old_png_file.with_suffix('.stdata')
            stdata_file.unlink(missing_ok=True)
            old_png_file.unlink()
