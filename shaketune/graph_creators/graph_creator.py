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
from datetime import datetime
from typing import Optional

from matplotlib.figure import Figure

from ..helpers.accelerometer import MeasurementsManager
from ..shaketune_config import ShakeTuneConfig


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

    def _save_figure(
        self, fig: Figure, measurements_manager: MeasurementsManager, axis_label: Optional[str] = None
    ) -> None:
        axis_suffix = f'_{axis_label}' if axis_label else ''
        filename = self._folder / f"{self._type.replace(' ', '')}_{self._graph_date}{axis_suffix}"
        fig.savefig(f'{filename}.png', dpi=self._config.dpi)

        if self._config.keep_raw_data:
            measurements_manager.save_stdata(f'{filename}.stdata')

    def get_type(self) -> str:
        return self._type

    @abc.abstractmethod
    def create_graph(self, measurements_manager: MeasurementsManager) -> None:
        pass

    def clean_old_files(self, keep_results: int = 3) -> None:
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)
        if len(files) <= keep_results:
            return  # No need to delete any files
        for old_png_file in files[keep_results:]:
            stdata_file = old_png_file.with_suffix('.stdata')
            stdata_file.unlink(missing_ok=True)
            old_png_file.unlink()
