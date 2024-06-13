# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: shaketune_process.py
# Description: Implements the ShakeTuneProcess class for managing the execution of
#              vibration analysis processes in separate system processes.


import multiprocessing
import os
import threading
import traceback
from typing import Optional

from .helpers.console_output import ConsoleOutput
from .shaketune_config import ShakeTuneConfig


class ShakeTuneProcess:
    def __init__(self, config: ShakeTuneConfig, graph_creator, timeout: Optional[float] = None) -> None:
        self._config = config
        self.graph_creator = graph_creator
        self._timeout = timeout

        self._process = None

    def get_graph_creator(self):
        return self.graph_creator

    def run(self) -> None:
        # Start the target function in a new process (a thread is known to cause issues with Klipper and CANbus due to the GIL)
        self._process = multiprocessing.Process(
            target=self._shaketune_process_wrapper, args=(self.graph_creator, self._timeout)
        )
        self._process.start()

    def wait_for_completion(self) -> None:
        if self._process is not None:
            self._process.join()

    # This function is a simple wrapper to start the Shake&Tune process. It's needed in order to get the timeout
    # as a Timer in a thread INSIDE the Shake&Tune child process to not interfere with the main Klipper process
    def _shaketune_process_wrapper(self, graph_creator, timeout) -> None:
        if timeout is not None:
            timer = threading.Timer(timeout, self._handle_timeout)
            timer.start()

        try:
            self._shaketune_process(graph_creator)
        finally:
            if timeout is not None:
                timer.cancel()

    def _handle_timeout(self) -> None:
        ConsoleOutput.print('Timeout: Shake&Tune computation did not finish within the specified timeout!')
        os._exit(1)  # Forcefully exit the process

    def _shaketune_process(self, graph_creator) -> None:
        # Trying to reduce Shake&Tune process priority to avoid slowing down the main Klipper process
        # as this could lead to random "Timer too close" errors when already running CANbus, etc...
        try:
            os.nice(19)
        except Exception:
            ConsoleOutput.print('Warning: failed reducing Shake&Tune process priority, continuing...')

        # Ensure the output folders exist
        for folder in self._config.get_results_subfolders():
            folder.mkdir(parents=True, exist_ok=True)

        # Generate the graphs
        try:
            graph_creator.create_graph()
        except FileNotFoundError as e:
            ConsoleOutput.print(f'FileNotFound error: {e}')
            return
        except TimeoutError as e:
            ConsoleOutput.print(f'Timeout error: {e}')
            return
        except Exception as e:
            ConsoleOutput.print(f'Error while generating the graphs: {e}\n{traceback.print_exc()}')
            return

        graph_creator.clean_old_files(self._config.keep_n_results)

        ConsoleOutput.print(f'{graph_creator.get_type()} graphs created successfully!')
        ConsoleOutput.print(
            f'Cleaned up the output folder (only the last {self._config.keep_n_results} results were kept)!'
        )
