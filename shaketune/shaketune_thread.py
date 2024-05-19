#!/usr/bin/env python3


import os
import threading
import traceback

from .helpers import filemanager as fm
from .helpers.console_output import ConsoleOutput
from .shaketune_config import ShakeTuneConfig


class ShakeTuneThread(threading.Thread):
    def __init__(self, config: ShakeTuneConfig, graph_creator, reactor, timeout: float):
        super(ShakeTuneThread, self).__init__()
        self._config = config
        self.graph_creator = graph_creator
        self._reactor = reactor
        self._timeout = timeout

    def get_graph_creator(self):
        return self.graph_creator

    def run(self) -> None:
        # Start the target function in a new thread
        internal_thread = threading.Thread(target=self._shaketune_thread, args=(self.graph_creator,))
        internal_thread.start()

        # Monitor the thread execution and stop it if it takes too long
        event_time = self._reactor.monotonic()
        end_time = event_time + self._timeout
        while event_time < end_time:
            event_time = self._reactor.pause(event_time + 0.05)
            if not internal_thread.is_alive():
                break

    # This function run in its own thread is used to do the CSV analysis and create the graphs
    def _shaketune_thread(self, graph_creator) -> None:
        # Trying to reduce the Shake&Tune prost-processing thread priority to avoid slowing down the main Klipper process
        # as this could lead to random "Timer" errors when already running CANbus, etc...
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
