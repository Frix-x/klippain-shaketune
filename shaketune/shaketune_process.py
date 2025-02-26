# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: shaketune_process.py
# Description: Implements the ShakeTuneProcess class for managing the execution of
#              vibration analysis processes in separate system processes.


import os
import threading
import traceback
from multiprocessing import Process
from pathlib import Path
from typing import List, Optional, Union

from .helpers.accelerometer import MeasurementsManager
from .helpers.console_output import ConsoleOutput
from .shaketune_config import ShakeTuneConfig


class ShakeTuneProcess:
    def __init__(self, st_config: ShakeTuneConfig, reactor, graph_creator, timeout: Optional[float] = None) -> None:
        self._config = st_config
        self._reactor = reactor
        self.graph_creator = graph_creator
        self._timeout = timeout
        self._process = None

    def get_graph_creator(self):
        return self.graph_creator

    def get_st_config(self):
        return self._config

    def run(self, filenames: Union[Path, List[Path]]) -> None:
        filelist = []

        # Single .stdata or a legacy .csv file
        if isinstance(filenames, Path):
            if filenames.suffix not in ('.stdata', '.csv'):
                filenames = filenames.with_suffix('.stdata')
            if not filenames.exists():
                raise FileNotFoundError(f'File {filenames} does not exist!')
            filelist.append(filenames)

        # List of legacy .csv files (still supported to be able to use CSVs when using the CLI mode)
        if isinstance(filenames, List):
            extensions = {f.suffix for f in filenames}
            if len(extensions) > 1:
                raise ValueError('Mixed file types are not allowed! Please provide Klipper CSV files only.')
            for filename in filenames:
                if filename.suffix != '.csv':
                    filename = filename.with_suffix('.csv')
                if not filename.exists():
                    raise FileNotFoundError(f'File {filename} does not exist!')
                filelist.append(filename)

        # Start the target function in a new process (a thread is known to cause issues with Klipper and CANbus due to the GIL)
        self._process = Process(
            target=self._shaketune_process_wrapper,
            args=(self.graph_creator, filelist, self._timeout),
        )
        self._process.start()

    def wait_for_completion(self) -> None:
        if self._process is None:
            return  # Nothing to wait for
        eventtime = self._reactor.monotonic()
        endtime = eventtime + self._timeout
        complete = False
        while eventtime < endtime:
            if not self._process.is_alive():
                complete = True
                break
            eventtime = self._reactor.pause(eventtime + 0.05)
        if not complete:
            self._handle_timeout()

    # This function is a simple wrapper to start the Shake&Tune process. It's needed in order to get the timeout
    # as a Timer in a thread INSIDE the Shake&Tune child process to not interfere with the main Klipper process
    def _shaketune_process_wrapper(self, graph_creator, filelist: List[Path], timeout) -> None:
        if timeout is not None:
            # Add 5 seconds to the timeout for safety. The goal is to avoid the Timer to finish before the
            # Shake&Tune process is done in case we call the wait_for_completion() function that uses Klipper's reactor.
            timeout += 5
            timer = threading.Timer(timeout, self._handle_timeout)
            timer.start()
        try:
            self._shaketune_process(graph_creator, filelist)
        finally:
            if timeout is not None:
                timer.cancel()

    def _handle_timeout(self) -> None:
        ConsoleOutput.print('Timeout: Shake&Tune computation did not finish within the specified timeout!')
        os._exit(1)  # Forcefully exit the process

    def _shaketune_process(self, graph_creator, filelist: List[Path]) -> None:
        # Reducing Shake&Tune process priority by putting the scheduler into batch mode with low priority. This in order to avoid
        # slowing down the main Klipper process as this can lead to random "Timer too close" or "Move queue overflow" errors
        # when also already running CANbus, neopixels and other consumming stuff in Klipper's main process.
        try:
            param = os.sched_param(os.sched_get_priority_min(os.SCHED_BATCH))
            os.sched_setscheduler(0, os.SCHED_BATCH, param)
        except Exception:
            ConsoleOutput.print('Warning: failed reducing Shake&Tune process priority, continuing...')

        # Load the measurements from the file
        m_manager = MeasurementsManager(self._config.chunk_size, self._reactor)
        if filelist[0].suffix == '.stdata':
            m_manager.load_from_stdata(filelist[0])
        else:
            m_manager.load_from_csvs(filelist)

        # Check if there are any measurements to process
        if m_manager.get_measurements() is None or len(m_manager.get_measurements()) == 0:
            ConsoleOutput.print('Error: no measurements available to create the graphs!')
            return

        # Generate the graphs
        try:
            graph_creator.create_graph(m_manager)
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
