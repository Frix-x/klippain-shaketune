#!/usr/bin/env python3

import abc
import re
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from matplotlib.figure import Figure

from ..helpers import filemanager as fm
from ..helpers.console_output import ConsoleOutput
from ..measurement.motorsconfigparser import MotorsConfigParser
from ..shaketune_config import ShakeTuneConfig
from .analyze_axesmap import axesmap_calibration
from .graph_belts import belts_calibration
from .graph_shaper import shaper_calibration
from .graph_vibrations import vibrations_profile


class GraphCreator(abc.ABC):
    def __init__(self, config: ShakeTuneConfig):
        self._config = config

        self._graph_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._version = ShakeTuneConfig.get_git_version()

        self._type = None
        self._folder = None

    def _setup_folder(self, graph_type: str) -> None:
        self._type = graph_type
        self._folder = self._config.get_results_folder(graph_type)

    def _move_and_prepare_files(
        self,
        glob_pattern: str,
        min_files_required: Optional[int] = None,
        custom_name_func: Optional[Callable[[Path], str]] = None,
    ) -> list[Path]:
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
            fm.wait_file_ready(filename)
            custom_name = custom_name_func(filename) if custom_name_func else filename.name
            new_file = self._folder / f'{self._type}_{self._graph_date}_{custom_name}.csv'
            # shutil.move() is needed to move the file across filesystems (mainly for BTT CB1 Pi default OS image)
            shutil.move(filename, new_file)
            lognames.append(new_file)
        return lognames

    def _save_figure_and_cleanup(self, fig: Figure, lognames: list[Path], axis_label: Optional[str] = None) -> None:
        axis_suffix = f'_{axis_label}' if axis_label else ''
        png_filename = self._folder / f'{self._type}_{self._graph_date}{axis_suffix}.png'
        fig.savefig(png_filename, dpi=self._config.dpi)

        if self._config.keep_csv:
            self._archive_files(lognames)
        else:
            self._remove_files(lognames)

    def _archive_files(self, _: list[Path]) -> None:
        return

    def _remove_files(self, lognames: list[Path]) -> None:
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


class BeltsGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config)

        self._kinematics = None
        self._accel_per_hz = None

        self._setup_folder('belts')

    def configure(self, kinematics: str = None, accel_per_hz: float = None) -> None:
        self._kinematics = kinematics
        self._accel_per_hz = accel_per_hz

    def create_graph(self) -> None:
        lognames = self._move_and_prepare_files(
            glob_pattern='shaketune-belt_*.csv',
            min_files_required=2,
            custom_name_func=lambda f: f.stem.split('_')[1].upper(),
        )
        fig = belts_calibration(
            lognames=[str(path) for path in lognames],
            kinematics=self._kinematics,
            klipperdir=str(self._config.klipper_folder),
            accel_per_hz=self._accel_per_hz,
            st_version=self._version,
        )
        self._save_figure_and_cleanup(fig, lognames)

    def clean_old_files(self, keep_results: int = 3) -> None:
        # Get all PNG files in the directory as a list of Path objects
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)

        if len(files) <= keep_results:
            return  # No need to delete any files

        # Delete the older files
        for old_file in files[keep_results:]:
            file_date = '_'.join(old_file.stem.split('_')[1:3])
            for suffix in ['A', 'B']:
                csv_file = self._folder / f'belts_{file_date}_{suffix}.csv'
                csv_file.unlink(missing_ok=True)
            old_file.unlink()


class ShaperGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config)

        self._max_smoothing = None
        self._scv = None

        self._setup_folder('shaper')

    def configure(self, scv: float, max_smoothing: float = None, accel_per_hz: float = None) -> None:
        self._scv = scv
        self._max_smoothing = max_smoothing
        self._accel_per_hz = accel_per_hz

    def create_graph(self) -> None:
        if not self._scv:
            raise ValueError('scv must be set to create the input shaper graph!')

        lognames = self._move_and_prepare_files(
            glob_pattern='shaketune-axis_*.csv',
            min_files_required=1,
            custom_name_func=lambda f: f.stem.split('_')[1].upper(),
        )
        fig = shaper_calibration(
            lognames=[str(path) for path in lognames],
            klipperdir=str(self._config.klipper_folder),
            max_smoothing=self._max_smoothing,
            scv=self._scv,
            accel_per_hz=self._accel_per_hz,
            st_version=self._version,
        )
        self._save_figure_and_cleanup(fig, lognames, lognames[0].stem.split('_')[-1])

    def clean_old_files(self, keep_results: int = 3) -> None:
        # Get all PNG files in the directory as a list of Path objects
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)

        if len(files) <= 2 * keep_results:
            return  # No need to delete any files

        # Delete the older files
        for old_file in files[2 * keep_results :]:
            csv_file = old_file.with_suffix('.csv')
            csv_file.unlink(missing_ok=True)
            old_file.unlink()


class VibrationsGraphCreator(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config)

        self._kinematics = None
        self._accel = None
        self._motors = None

        self._setup_folder('vibrations')

    def configure(self, kinematics: str, accel: float, motor_config_parser: MotorsConfigParser) -> None:
        self._kinematics = kinematics
        self._accel = accel
        self._motors = motor_config_parser.get_motors()

    def _archive_files(self, lognames: list[Path]) -> None:
        tar_path = self._folder / f'{self._type}_{self._graph_date}.tar.gz'
        with tarfile.open(tar_path, 'w:gz') as tar:
            for csv_file in lognames:
                tar.add(csv_file, arcname=csv_file.name, recursive=False)
                csv_file.unlink()

    def create_graph(self) -> None:
        if not self._accel or not self._kinematics:
            raise ValueError('accel, chip_name and kinematics must be set to create the vibrations profile graph!')

        lognames = self._move_and_prepare_files(
            glob_pattern='shaketune-vib_*.csv',
            min_files_required=None,
            custom_name_func=lambda f: re.search(r'shaketune-vib_(.*?)_\d{8}_\d{6}', f.name).group(1),
        )
        fig = vibrations_profile(
            lognames=[str(path) for path in lognames],
            klipperdir=str(self._config.klipper_folder),
            kinematics=self._kinematics,
            accel=self._accel,
            st_version=self._version,
            motors=self._motors,
        )
        self._save_figure_and_cleanup(fig, lognames)

    def clean_old_files(self, keep_results: int = 3) -> None:
        # Get all PNG files in the directory as a list of Path objects
        files = sorted(self._folder.glob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)

        if len(files) <= keep_results:
            return  # No need to delete any files

        # Delete the older files
        for old_file in files[keep_results:]:
            old_file.unlink()
            tar_file = old_file.with_suffix('.tar.gz')
            tar_file.unlink(missing_ok=True)


class AxesMapFinder(GraphCreator):
    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config)

        self._graph_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._type = 'axesmap'
        self._folder = config.get_results_folder()

        self._accel = None

    def configure(self, accel: int) -> None:
        self._accel = accel

    def find_axesmap(self) -> None:
        tmp_folder = Path('/tmp')
        globbed_files = list(tmp_folder.glob('shaketune-axemap_*.csv'))

        if not globbed_files:
            raise FileNotFoundError('no CSV files found in the /tmp folder to find the axes map!')

        # Find the CSV files with the latest timestamp and process it
        logname = sorted(globbed_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        results = axesmap_calibration(
            lognames=[str(logname)],
            accel=self._accel,
        )
        ConsoleOutput.print(results)

        result_filename = self._folder / f'{self._type}_{self._graph_date}.txt'
        with result_filename.open('w') as f:
            f.write(results)

    # While the AxesMapFinder doesn't directly create a graph, we need to implement this
    # method to allow using it seemlessly like all the other GraphCreator objects
    def create_graph(self) -> None:
        self.find_axesmap()

    def clean_old_files(self, keep_results: int) -> None:
        tmp_folder = Path('/tmp')
        globbed_files = list(tmp_folder.glob('shaketune-axemap_*.csv'))
        for csv_file in globbed_files:
            csv_file.unlink()
