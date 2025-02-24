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
from pathlib import Path

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
        self._version = ShakeTuneConfig.get_git_version()
        self._type = self.__class__.graph_type
        self._folder = self._config.get_results_folder(self._type)
        self._plotter = Plotter()
        self._output_target: Path = None

    def _save_figure(self, fig: Figure) -> None:
        if self._output_target is None:
            raise ValueError(
                'Output target not defined. Please call define_output_target() before trying to save the figure!'
            )

        fig.savefig(f'{self._output_target.with_suffix(".png")}', dpi=self._config.dpi)
        if not self._config.keep_raw_data:
            self._output_target.with_suffix('.stdata').unlink(missing_ok=True)

    def get_type(self) -> str:
        return self._type

    def get_folder(self) -> Path:
        return self._folder

    def define_output_target(self, filepath: Path) -> None:
        # Remove the extension if it exists (to be safer when using the CLI mode)
        if filepath.suffix:
            filepath = filepath.with_suffix('')
        self._output_target = filepath

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
