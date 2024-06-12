# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: shaketune_config.py
# Description: Defines the ShakeTuneConfig class for handling configuration settings
#              and file paths related to Shake&Tune operations.


from pathlib import Path

from .helpers.console_output import ConsoleOutput

KLIPPER_FOLDER = Path.home() / 'klipper'
KLIPPER_LOG_FOLDER = Path.home() / 'printer_data/logs'
RESULTS_BASE_FOLDER = Path.home() / 'printer_data/config/K-ShakeTune_results'
RESULTS_SUBFOLDERS = {
    'axes map': 'axes_map',
    'belts comparison': 'belts',
    'input shaper': 'input_shaper',
    'vibrations profile': 'vibrations',
    'static frequency': 'static_freq',
}


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
