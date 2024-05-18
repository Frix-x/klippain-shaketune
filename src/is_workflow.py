#!/usr/bin/env python3

############################################
###### INPUT SHAPER KLIPPAIN WORKFLOW ######
############################################
# Written by Frix_x#0161 #

#   This script is designed to be used with gcode_shell_commands directly from Klipper
#   Use the provided Shake&Tune macros instead!


import abc
import argparse
import shutil
import tarfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from git import GitCommandError, Repo
from matplotlib.figure import Figure

import src.helpers.filemanager as fm
from src.graph_creators.analyze_axesmap import axesmap_calibration
from src.graph_creators.graph_belts import belts_calibration
from src.graph_creators.graph_shaper import shaper_calibration
from src.graph_creators.graph_vibrations import vibrations_profile
from src.helpers.locale_utils import print_with_c_locale
from src.helpers.motorlogparser import MotorLogParser


class Config:
    KLIPPER_FOLDER = Path.home() / 'klipper'
    KLIPPER_LOG_FOLDER = Path.home() / 'printer_data/logs'
    RESULTS_BASE_FOLDER = Path.home() / 'printer_data/config/K-ShakeTune_results'
    RESULTS_SUBFOLDERS = {'belts': 'belts', 'shaper': 'inputshaper', 'vibrations': 'vibrations'}

    @staticmethod
    def get_results_folder(type: str) -> Path:
        return Config.RESULTS_BASE_FOLDER / Config.RESULTS_SUBFOLDERS[type]

    @staticmethod
    def get_git_version() -> str:
        try:
            # Get the absolute path of the script, resolving any symlinks
            # Then get 1 times to parent dir to be at the git root folder
            script_path = Path(__file__).resolve()
            repo_path = script_path.parents[1]
            repo = Repo(repo_path)
            try:
                version = repo.git.describe('--tags')
            except GitCommandError:
                version = repo.head.commit.hexsha[:7]  # If no tag is found, use the simplified commit SHA instead
            return version
        except Exception as e:
            print_with_c_locale(f'Warning: unable to retrieve Shake&Tune version number: {e}')
            return 'unknown'

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Shake&Tune graphs generation script')
        parser.add_argument(
            '-t',
            '--type',
            dest='type',
            choices=['belts', 'shaper', 'vibrations', 'axesmap'],
            required=True,
            help='Type of output graph to produce',
        )
        parser.add_argument(
            '--accel',
            type=int,
            default=None,
            dest='accel_used',
            help='Accelerometion used for vibrations profile creation or axes map calibration',
        )
        parser.add_argument(
            '--chip_name',
            type=str,
            default='adxl345',
            dest='chip_name',
            help='Accelerometer chip name used for vibrations profile creation or axes map calibration',
        )
        parser.add_argument(
            '--max_smoothing',
            type=float,
            default=None,
            dest='max_smoothing',
            help='Maximum smoothing to allow for input shaper filter recommendations',
        )
        parser.add_argument(
            '--scv',
            '--square_corner_velocity',
            type=float,
            default=5.0,
            dest='scv',
            help='Square corner velocity used to compute max accel for input shapers filter recommendations',
        )
        parser.add_argument(
            '-m',
            '--kinematics',
            dest='kinematics',
            default='cartesian',
            choices=['cartesian', 'corexy'],
            help='Machine kinematics configuration used for the vibrations profile creation',
        )
        parser.add_argument(
            '--metadata',
            type=str,
            default=None,
            dest='metadata',
            help='Motor configuration metadata printed on the vibrations profiles',
        )
        parser.add_argument(
            '-c',
            '--keep_csv',
            action='store_true',
            default=False,
            dest='keep_csv',
            help='Whether to keep the raw CSV files after processing in addition to the PNG graphs',
        )
        parser.add_argument(
            '-n',
            '--keep_results',
            type=int,
            default=3,
            dest='keep_results',
            help='Number of results to keep in the result folder after each run of the script',
        )
        parser.add_argument('--dpi', type=int, default=150, dest='dpi', help='DPI of the output PNG files')
        parser.add_argument('-v', '--version', action='version', version=f'Shake&Tune {Config.get_git_version()}')

        return parser.parse_args()


class GraphCreator(abc.ABC):
    def __init__(self, keep_csv: bool, dpi: int):
        self._keep_csv = keep_csv
        self._dpi = dpi

        self._graph_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._version = Config.get_git_version()

        self._type = None
        self._folder = None

    def _setup_folder(self, graph_type: str) -> None:
        self._type = graph_type
        self._folder = Config.get_results_folder(graph_type)

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
            fm.wait_file_ready(new_file)
            lognames.append(new_file)
        return lognames

    def _save_figure_and_cleanup(self, fig: Figure, lognames: list[Path], axis_label: Optional[str] = None) -> None:
        axis_suffix = f'_{axis_label}' if axis_label else ''
        png_filename = self._folder / f'{self._type}_{self._graph_date}{axis_suffix}.png'
        fig.savefig(png_filename, dpi=self._dpi)

        if self._keep_csv:
            self._archive_files(lognames)
        else:
            self._remove_files(lognames)

    def _archive_files(self, _: list[Path]) -> None:
        return

    def _remove_files(self, lognames: list[Path]) -> None:
        for csv in lognames:
            csv.unlink(missing_ok=True)

    @abc.abstractmethod
    def create_graph(self) -> None:
        pass

    @abc.abstractmethod
    def clean_old_files(self, keep_results: int) -> None:
        pass


class BeltsGraphCreator(GraphCreator):
    def __init__(self, keep_csv: bool = False, dpi: int = 150):
        super().__init__(keep_csv, dpi)

        self._setup_folder('belts')

    def create_graph(self) -> None:
        lognames = self._move_and_prepare_files(
            glob_pattern='raw_data_axis*.csv',
            min_files_required=2,
            custom_name_func=lambda f: f.stem.split('_')[3].upper(),
        )
        fig = belts_calibration(
            lognames=[str(path) for path in lognames],
            klipperdir=str(Config.KLIPPER_FOLDER),
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
    def __init__(self, keep_csv: bool = False, dpi: int = 150):
        super().__init__(keep_csv, dpi)

        self._max_smoothing = None
        self._scv = None

        self._setup_folder('shaper')

    def configure(self, scv: float, max_smoothing: float = None) -> None:
        self._scv = scv
        self._max_smoothing = max_smoothing

    def create_graph(self) -> None:
        if not self._scv:
            raise ValueError('scv must be set to create the input shaper graph!')

        lognames = self._move_and_prepare_files(
            glob_pattern='raw_data*.csv',
            min_files_required=1,
            custom_name_func=lambda f: f.stem.split('_')[3].upper(),
        )
        fig = shaper_calibration(
            lognames=[str(path) for path in lognames],
            klipperdir=str(Config.KLIPPER_FOLDER),
            max_smoothing=self._max_smoothing,
            scv=self._scv,
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
    def __init__(self, keep_csv: bool = False, dpi: int = 150):
        super().__init__(keep_csv, dpi)

        self._kinematics = None
        self._accel = None
        self._chip_name = None
        self._motors = None

        self._setup_folder('vibrations')

    def configure(self, kinematics: str, accel: float, chip_name: str, metadata: str) -> None:
        self._kinematics = kinematics
        self._accel = accel
        self._chip_name = chip_name

        parser = MotorLogParser(Config.KLIPPER_LOG_FOLDER / 'klippy.log', metadata)
        self._motors = parser.get_motors()

    def _archive_files(self, lognames: list[Path]) -> None:
        tar_path = self._folder / f'{self._type}_{self._graph_date}.tar.gz'
        with tarfile.open(tar_path, 'w:gz') as tar:
            for csv_file in lognames:
                tar.add(csv_file, arcname=csv_file.name, recursive=False)

    def create_graph(self) -> None:
        if not self._accel or not self._chip_name or not self._kinematics:
            raise ValueError('accel, chip_name and kinematics must be set to create the vibrations profile graph!')

        lognames = self._move_and_prepare_files(
            glob_pattern=f'{self._chip_name}-*.csv',
            min_files_required=None,
            custom_name_func=lambda f: f.name.replace(self._chip_name, self._type),
        )
        fig = vibrations_profile(
            lognames=[str(path) for path in lognames],
            klipperdir=str(Config.KLIPPER_FOLDER),
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


class AxesMapFinder:
    def __init__(self, accel: float, chip_name: str):
        self._accel = accel
        self._chip_name = chip_name

        self._graph_date = datetime.now().strftime('%Y%m%d_%H%M%S')

        self._type = 'axesmap'
        self._folder = Config.RESULTS_BASE_FOLDER

    def find_axesmap(self) -> None:
        tmp_folder = Path('/tmp')
        globbed_files = list(tmp_folder.glob(f'{self._chip_name}-*.csv'))

        if not globbed_files:
            raise FileNotFoundError('no CSV files found in the /tmp folder to find the axes map!')

        # Find the CSV files with the latest timestamp and wait for it to be released by Klipper
        logname = sorted(globbed_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        fm.wait_file_ready(logname)

        results = axesmap_calibration(
            lognames=[str(logname)],
            accel=self._accel,
        )

        result_filename = self._folder / f'{self._type}_{self._graph_date}.txt'
        with result_filename.open('w') as f:
            f.write(results)


def main():
    options = Config.parse_arguments()
    fm.ensure_folders_exist(
        folders=[Config.RESULTS_BASE_FOLDER / subfolder for subfolder in Config.RESULTS_SUBFOLDERS.values()]
    )

    print_with_c_locale(f'Shake&Tune version: {Config.get_git_version()}')

    graph_creators = {
        'belts': (BeltsGraphCreator, None),
        'shaper': (ShaperGraphCreator, lambda gc: gc.configure(options.scv, options.max_smoothing)),
        'vibrations': (
            VibrationsGraphCreator,
            lambda gc: gc.configure(options.kinematics, options.accel_used, options.chip_name, options.metadata),
        ),
        'axesmap': (AxesMapFinder, None),
    }

    creator_info = graph_creators.get(options.type)
    if not creator_info:
        print_with_c_locale('Error: invalid graph type specified!')
        return

    # Instantiate the graph creator
    graph_creator_class, configure_func = creator_info
    graph_creator = graph_creator_class(options.keep_csv, options.dpi)

    # Configure it if needed
    if configure_func:
        configure_func(graph_creator)

    # And then run it
    try:
        graph_creator.create_graph()
    except FileNotFoundError as e:
        print_with_c_locale(f'FileNotFound error: {e}')
        return
    except TimeoutError as e:
        print_with_c_locale(f'Timeout error: {e}')
        return
    except Exception as e:
        print_with_c_locale(f'Error while generating the graphs: {e}')
        traceback.print_exc()
        return

    print_with_c_locale(f'{options.type} graphs created successfully!')
    graph_creator.clean_old_files(options.keep_results)
    print_with_c_locale(f'Cleaned output folder to keep only the last {options.keep_results} results!')


if __name__ == '__main__':
    main()
