#!/usr/bin/env python3

from pathlib import Path

from .helpers.console_output import ConsoleOutput

KLIPPER_FOLDER = Path.home() / 'klipper'
KLIPPER_LOG_FOLDER = Path.home() / 'printer_data/logs'
RESULTS_BASE_FOLDER = Path.home() / 'printer_data/config/K-ShakeTune_results'
RESULTS_SUBFOLDERS = {'belts': 'belts', 'shaper': 'inputshaper', 'vibrations': 'vibrations'}


class ShakeTuneConfig:
    def __init__(
        self, result_folder: Path = RESULTS_BASE_FOLDER, keep_n_results: int = 3, keep_csv: bool = False, dpi: int = 150
    ) -> None:
        self._result_folder = result_folder

        self.keep_n_results = keep_n_results
        self.keep_csv = keep_csv
        self.dpi = dpi

        self.klipper_folder = KLIPPER_FOLDER
        self.klipper_log_folder = KLIPPER_LOG_FOLDER

    def get_results_folder(self, type: str = None) -> Path:
        if type is None:
            return self._result_folder
        else:
            return self._result_folder / RESULTS_SUBFOLDERS[type]

    def get_results_subfolders(self) -> Path:
        subfolders = [self._result_folder / subfolder for subfolder in RESULTS_SUBFOLDERS.values()]
        return subfolders

    @staticmethod
    def get_git_version() -> str:
        try:
            from git import GitCommandError, Repo

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
            ConsoleOutput.print(f'Warning: unable to retrieve Shake&Tune version number: {e}')
            return 'unknown'

    # @staticmethod
    # def parse_arguments(params: Optional[List] = None) -> argparse.Namespace:
    #     parser = argparse.ArgumentParser(description='Shake&Tune graphs generation script')
    #     parser.add_argument(
    #         '-t',
    #         '--type',
    #         dest='type',
    #         choices=['belts', 'shaper', 'vibrations', 'axesmap'],
    #         required=True,
    #         help='Type of output graph to produce',
    #     )
    #     parser.add_argument(
    #         '--accel',
    #         type=int,
    #         default=None,
    #         dest='accel_used',
    #         help='Accelerometion used for vibrations profile creation or axes map calibration',
    #     )
    #     parser.add_argument(
    #         '--chip_name',
    #         type=str,
    #         default='adxl345',
    #         dest='chip_name',
    #         help='Accelerometer chip name used for vibrations profile creation or axes map calibration',
    #     )
    #     parser.add_argument(
    #         '--max_smoothing',
    #         type=float,
    #         default=None,
    #         dest='max_smoothing',
    #         help='Maximum smoothing to allow for input shaper filter recommendations',
    #     )
    #     parser.add_argument(
    #         '--scv',
    #         '--square_corner_velocity',
    #         type=float,
    #         default=5.0,
    #         dest='scv',
    #         help='Square corner velocity used to compute max accel for input shapers filter recommendations',
    #     )
    #     parser.add_argument(
    #         '-m',
    #         '--kinematics',
    #         dest='kinematics',
    #         default='cartesian',
    #         choices=['cartesian', 'corexy'],
    #         help='Machine kinematics configuration used for the vibrations profile creation',
    #     )
    #     parser.add_argument(
    #         '--metadata',
    #         type=str,
    #         default=None,
    #         dest='metadata',
    #         help='Motor configuration metadata printed on the vibrations profiles',
    #     )
    #     parser.add_argument(
    #         '-c',
    #         '--keep_csv',
    #         action='store_true',
    #         default=False,
    #         dest='keep_csv',
    #         help='Whether to keep the raw CSV files after processing in addition to the PNG graphs',
    #     )
    #     parser.add_argument(
    #         '-n',
    #         '--keep_results',
    #         type=int,
    #         default=3,
    #         dest='keep_results',
    #         help='Number of results to keep in the result folder after each run of the script',
    #     )
    #     parser.add_argument('--dpi', type=int, default=150, dest='dpi', help='DPI of the output PNG files')
    #     parser.add_argument(
    #         '-v', '--version', action='version', version=f'Shake&Tune {ShakeTuneConfig.get_git_version()}'
    #     )

    #     return parser.parse_args(params)
