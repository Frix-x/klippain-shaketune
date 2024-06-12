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
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from matplotlib.figure import Figure

from ..shaketune_config import ShakeTuneConfig


class GraphCreator(abc.ABC):
    def __init__(self, config: ShakeTuneConfig, graph_type: str):
        self._config = config
        self._graph_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._version = ShakeTuneConfig.get_git_version()
        self._type = graph_type
        self._folder = self._config.get_results_folder(graph_type)

    def _move_and_prepare_files(
        self,
        glob_pattern: str,
        min_files_required: Optional[int] = None,
        custom_name_func: Optional[Callable[[Path], str]] = None,
    ) -> List[Path]:
        tmp_path = Path('/tmp')
        globbed_files = list(tmp_path.glob(glob_pattern))

        # If min_files_required is not set, use the number of globbed files as the minimum
        min_files_required = min_files_required or len(globbed_files)

        if not globbed_files:
            raise FileNotFoundError(f'no CSV files found in the /tmp folder to create the {self._type} graphs!')
        if len(globbed_files) < min_files_required:
            raise FileNotFoundError(f'{min_files_required} CSV files are needed to create the {self._type} graphs!')

        lognames = []
        for filename in sorted(globbed_files, key=lambda f: f.stat().st_mtime, reverse=True)[:min_files_required]:
            custom_name = custom_name_func(filename) if custom_name_func else filename.name
            new_file = self._folder / f"{self._type.replace(' ', '')}_{self._graph_date}_{custom_name}.csv"
            # shutil.move() is needed to move the file across filesystems (mainly for BTT CB1 Pi default OS image)
            shutil.move(filename, new_file)
            lognames.append(new_file)
        return lognames

    def _save_figure_and_cleanup(self, fig: Figure, lognames: List[Path], axis_label: Optional[str] = None) -> None:
        axis_suffix = f'_{axis_label}' if axis_label else ''
        png_filename = self._folder / f"{self._type.replace(' ', '')}_{self._graph_date}{axis_suffix}.png"
        fig.savefig(png_filename, dpi=self._config.dpi)

        if self._config.keep_csv:
            self._archive_files(lognames)
        else:
            self._remove_files(lognames)

    def _archive_files(self, lognames: List[Path]) -> None:
        return

    def _remove_files(self, lognames: List[Path]) -> None:
        for csv in lognames:
            csv.unlink(missing_ok=True)

    def get_type(self) -> str:
        return self._type

    @abc.abstractmethod
    def create_graph(self) -> None:
        pass

    @abc.abstractmethod
    def clean_old_files(self, keep_results: int) -> None:
        pass
