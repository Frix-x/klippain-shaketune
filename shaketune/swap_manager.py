# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2022 - 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: swap_manager.py
# Description: Implements the SwapManager class for managing the creation and
#              activation of a temporary swap file on the system to avoid running
#              out of memory when processing large files (useful for low end devices like CB1)

import shutil
import subprocess
from pathlib import Path

from .helpers.console_output import ConsoleOutput

SWAP_FILE_PATH = Path.home() / 'shaketune_swap'


class SwapManager:
    def __init__(self, swap_size_mb: int = 0) -> None:
        self._swap_size_mb = swap_size_mb
        self._swap_file_path = SWAP_FILE_PATH
        self._swap_activated = False

    def is_swap_activated(self) -> bool:
        return self._swap_activated

    def add_swap(self) -> None:
        if self._swap_size_mb <= 0:
            return

        # Check if swap file already exists and delete it if it does
        if self._swap_file_path.exists():
            ConsoleOutput.print(f'Warning: {self._swap_file_path} already exists. Replacing it...')
            self.remove_swap()

        # Check available disk space to be sure there is enough space for the swap file
        total, used, free = shutil.disk_usage(self._swap_file_path.parent)
        free_mb = free // (1024 * 1024)
        if free_mb < self._swap_size_mb:
            ConsoleOutput.print(
                f'Warning: not enough disk space available ({free_mb} MB) to create the temporary swap file '
                f'that you asked for ({self._swap_size_mb} MB). It will not be created for this run...'
            )
            return

        # Create the swap file and activate it
        try:
            fallocate_result = subprocess.run(
                [
                    'sudo',
                    'fallocate',
                    '-l',
                    f'{self._swap_size_mb}M',
                    str(self._swap_file_path),
                ],
                check=True,
            )
            if fallocate_result.returncode != 0:
                ConsoleOutput.print(
                    'Warning: failed to allocate the temporary swap file using fallocate. '
                    'Falling back to dd, this may take a while...'
                )
                subprocess.run(
                    [
                        'sudo',
                        'dd',
                        'if=/dev/zero',
                        f'of={self._swap_file_path}',
                        'bs=1M',
                        f'count={self._swap_size_mb}',
                    ],
                    check=True,
                )
            subprocess.run(['sudo', 'chmod', '600', str(self._swap_file_path)], check=True)
            subprocess.run(['sudo', 'mkswap', str(self._swap_file_path)], check=True)
            subprocess.run(['sudo', 'swapon', str(self._swap_file_path)], check=True)
            self._swap_activated = True
            ConsoleOutput.print(f'Temporary swap file of {self._swap_size_mb} MB activated')
        except subprocess.CalledProcessError as err:
            self.remove_swap()
            raise RuntimeError('Failed to create and activate the temporary swap file!') from err

    def remove_swap(self) -> None:
        if not self._swap_file_path.exists():
            return

        try:
            if self._swap_activated:
                subprocess.run(['sudo', 'swapoff', str(self._swap_file_path)], check=True)
            subprocess.run(['sudo', 'rm', str(self._swap_file_path)], check=True)
        except subprocess.CalledProcessError as err:
            raise RuntimeError('Failed to deactivate and delete the temporary swap file!') from err
