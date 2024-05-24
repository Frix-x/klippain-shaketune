#!/usr/bin/env python3


import os
import threading
import traceback
from typing import Optional

from .helpers import filemanager as fm
from .helpers.console_output import ConsoleOutput
from .shaketune_config import ShakeTuneConfig


class ShakeTuneThread(threading.Thread):
    def __init__(self, config: ShakeTuneConfig, graph_creator, timeout: Optional[float] = None) -> None:
        super(ShakeTuneThread, self).__init__()
        self._config = config
        self.graph_creator = graph_creator
        self._timeout = timeout

        self._internal_thread = None
        self._stop_event = threading.Event()

    def get_graph_creator(self):
        return self.graph_creator

    def run(self) -> None:
        # Start the target function in a new thread
        self._internal_thread = threading.Thread(target=self._shaketune_thread, args=(self.graph_creator,))
        self._internal_thread.start()

        # If a timeout is specified, start a timer thread to monitor the timeout
        if self._timeout is not None:
            timer_thread = threading.Timer(self._timeout, self._handle_timeout)
            timer_thread.start()

    def _handle_timeout(self) -> None:
        if self._internal_thread.is_alive():
            self._stop_event.set()
            ConsoleOutput.print('Timeout: Shake&Tune computation did not finish within the specified timeout!')

    def wait_for_completion(self) -> None:
        if self._internal_thread is not None:
            self._internal_thread.join()

    # This function run in a thread is used to do the CSV analysis and create the graphs
    def _shaketune_thread(self, graph_creator) -> None:
        # Trying to reduce the Shake&Tune post-processing thread priority to avoid slowing down the main Klipper process
        # as this could lead to random "Timer too close" errors when already running CANbus, etc...
        try:
            os.nice(20)
        except Exception:
            ConsoleOutput.print('Warning: failed reducing Shake&Tune thread priority, continuing...')

        fm.ensure_folders_exist(self._config.get_results_subfolders())

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

        if graph_creator.get_type() != 'axesmap':
            ConsoleOutput.print(f'{graph_creator.get_type()} graphs created successfully!')
            ConsoleOutput.print(
                f'Cleaned up the output folder (only the last {self._config.keep_n_results} results were kept)!'
            )
