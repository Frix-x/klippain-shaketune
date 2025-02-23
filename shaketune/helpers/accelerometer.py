# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: accelerometer.py
# Description: Provides a custom and internal Shake&Tune Accelerometer helper that interfaces
#              with Klipper's accelerometer classes. It includes functions to start and stop
#              accelerometer measurements.
#              It also includes functions to load and save measurements from a file in a new
#              compressed format (.stdata) or from the legacy Klipper CSV files.


import json
import time
import uuid
from io import TextIOWrapper
from multiprocessing import Process, Queue, Value
from pathlib import Path
from shutil import move as move_file
from typing import List, Optional, Tuple, TypedDict

import numpy as np
from zstandard import FLUSH_FRAME, ZstdCompressor, ZstdDecompressor

from ..helpers.console_output import ConsoleOutput

Sample = Tuple[float, float, float, float]
SamplesList = List[Sample]

STOP_SENTINEL = 'STOP_SENTINEL'
WRITE_TIMEOUT = 30
WAIT_FOR_SAMPLE_TIMEOUT = 30


class Measurement(TypedDict):
    name: str
    samples: SamplesList


class MeasurementsManager:
    def __init__(self, chunk_size: int, k_reactor=None):
        # Klipper reactor is optional here as when running in CLI mode, we don't need it since in this mode
        # we are only reading a file to create graphs and never recording anything (so no disk writes)
        self._k_reactor = k_reactor
        self._chunk_size = chunk_size

        self.measurements: List[Measurement] = []
        self._temp_file = Path(f'/tmp/shaketune_{str(uuid.uuid4())[:8]}.stdata')

        # Create a dedicated process with a Queue to manage all the writing operations
        self._writer_queue = Queue()
        self._is_writing = Value('b', False)
        self._writer_process: Optional[Process] = None

    # Dedicated writer process: opens the output file in binary write mode and wraps it with a Zstandard compressor
    # stream. It then continuously reads measurement objects from the queue and writes each as a JSON line
    def _writer_loop(self, output_file: Path, write_queue: Queue, is_writing: Value):
        try:
            with open(output_file, 'wb') as f:
                cctx = ZstdCompressor(level=3)
                with cctx.stream_writer(f) as compressor:
                    while True:
                        meas = write_queue.get()
                        if meas == STOP_SENTINEL:
                            break
                        with is_writing.get_lock():
                            is_writing.value = True
                        line = (json.dumps(meas) + '\n').encode('utf-8')
                        compressor.write(line)
                        with is_writing.get_lock():
                            is_writing.value = False
                    compressor.flush(FLUSH_FRAME)
        except Exception as e:
            ConsoleOutput.print(f'Error writing to file {output_file}: {e}')

    def clear_measurements(self, keep_last: bool = False):
        self.measurements = [self.measurements[-1]] if keep_last and self.measurements else []

    def append_samples_to_current_measurement(self, additional_samples: SamplesList):
        try:
            self.measurements[-1]['samples'].extend(additional_samples)
        except IndexError as err:
            raise ValueError('no measurements available to append samples to!') from err

    def add_measurement(self, name: str, samples: SamplesList = None, timeout: float = WRITE_TIMEOUT):
        # Start the writer process if it's not already running
        if self._writer_process is None:
            self._writer_process = Process(
                target=self._writer_loop,
                args=(self._temp_file, self._writer_queue, self._is_writing),
                daemon=False,
            )
            self._writer_process.start()

        samples = samples if samples is not None else []
        self.measurements.append({'name': name, 'samples': samples})
        if len(self.measurements) > self._chunk_size:
            self._flush_chunk()  # Flush the current chunk of measurements to disk

            # Force wait for the writer process to finish writing in order to avoid being able
            # to start a new measurement while the previous one is still being written on disk
            # This is necessary to avoid Timer too close errors in Klipper...
            if self._k_reactor is None:
                return  # In case no reactor is available, we can't wait for the writer to finish
            eventtime = self._k_reactor.monotonic()
            endtime = eventtime + timeout
            while eventtime < endtime:
                with self._is_writing.get_lock():
                    if self._writer_queue.empty() and not self._is_writing.value:
                        return  # writer process is idle, so we can continue...
                eventtime = self._k_reactor.pause(eventtime + 0.05)

            # Raise an error with some statistics about the writer process
            raise TimeoutError(
                'Timeout while waiting for the writer process to finish writing measurements chunk to disk!\n'
                f'Writer process is still running and has {self._writer_queue.qsize()} items in the queue!'
            )

    # Flush all measurements except the last one (which can still receive appended samples) to the dedicated
    # writer process. Each measurement is sent as a single JSON-serializable object via the Queue
    def _flush_chunk(self):
        if len(self.measurements) <= 1:
            return
        flush_list = self.measurements[:-1]
        for meas in flush_list:
            self._writer_queue.put(meas)
        self.clear_measurements(keep_last=True)

    def save_stdata(self, filename: Path, timeout: int = WRITE_TIMEOUT):
        # Klipper reactor is required to save the data to disk (but optional for the CLI mode that never saves any .stdata)
        if not self._k_reactor:
            raise ValueError('No Klipper reactor provided! Unable to save data to disk.')

        if not self._writer_process:
            raise ValueError('No writer process available! Unable to save data to disk.')

        # Add extension if not provided
        if filename.suffix != '.stdata':
            filename = filename.with_suffix('.stdata')

        # Flush any remaining in-memory measurements
        if len(self.measurements) > 0:
            for meas in self.measurements:
                self._writer_queue.put(meas)
            self.clear_measurements()

        # Signal the writer process to finish its task
        self._writer_queue.put(STOP_SENTINEL)

        # Wait for the writer process to finish its task
        eventtime = self._k_reactor.monotonic()
        endtime = eventtime + timeout
        complete = False
        while eventtime < endtime:
            if not self._writer_process.is_alive():
                complete = True
                break
            eventtime = self._k_reactor.pause(eventtime + 0.05)
        if not complete:
            raise TimeoutError(
                'Shake&Tune was unable to finish and close the .stdata writing process. '
                'This might be due to a slow, busy or full SD card.'
            )

        try:
            if filename.exists():
                filename.unlink()
            move_file(self._temp_file, filename)  # using shutil.move() to avoid cross-filesystem issues
        except Exception as e:
            ConsoleOutput.print(f'Shake&Tune was unable to create the final data file ({filename}): {e}')

    # Return all the measurements from memory. Measurements flushed to disk are available via load_from_stdata()
    def get_measurements(self) -> List[Measurement]:
        return self.measurements

    # Load all the measurements from the .stdata file
    def load_from_stdata(self, filename: Path) -> List[Measurement]:
        measurements = []
        try:
            with open(filename, 'rb') as f:
                dctx = ZstdDecompressor()
                with dctx.stream_reader(f) as decompressor:
                    text_stream = TextIOWrapper(decompressor, encoding='utf-8')
                    for line in text_stream:
                        if line.strip():
                            meas = json.loads(line)
                            measurements.append(meas)
            self.measurements = measurements
        except Exception as e:
            ConsoleOutput.print(f'Warning: unable to load measurements from {filename}: {e}')
            self.measurements = []

    def load_from_csvs(self, klipper_CSVs: List[Path]) -> List[Measurement]:
        for logname in klipper_CSVs:
            try:
                if logname.suffix != '.csv':
                    ConsoleOutput.print(f'Warning: {logname} is not a CSV file. It will be ignored by Shake&Tune!')
                    continue
                with open(logname) as f:
                    header = None
                    for line in f:
                        cleaned_line = line.strip()
                        # Check for a PSD file generated by Klipper and raise a warning
                        if cleaned_line.startswith('#freq,psd_x,psd_y,psd_z,psd_xyz'):
                            ConsoleOutput.print(
                                f'Warning: {logname} does not contain raw Klipper accelerometer data. '
                                'Please use the official Klipper script to process it instead. '
                            )
                            continue
                        # Check for the expected legacy header used in Shake&Tune (raw accelerometer data from Klipper)
                        elif cleaned_line.startswith('#time,accel_x,accel_y,accel_z'):
                            header = cleaned_line
                            break
                    if not header:
                        ConsoleOutput.print(
                            f"Warning: file {logname} doesn't seem to be a Klipper raw accelerometer data file. "
                            f"Expected '#time,accel_x,accel_y,accel_z', but got '{header.strip()}'. "
                            'This file will be ignored by Shake&Tune!'
                        )
                        continue
                    # If we have the correct raw data header, proceed to load the data
                    data = np.loadtxt(logname, comments='#', delimiter=',', skiprows=1)
                    if data.ndim == 1 or data.shape[1] != 4:
                        ConsoleOutput.print(
                            f'Warning: {logname} does not have the correct data format; expected 4 columns. '
                            'It will be ignored by Shake&Tune!'
                        )
                        continue

                    # Add the parsed klipper raw accelerometer data to Shake&Tune measurements object
                    samples = [tuple(row) for row in data]
                    self.add_measurement(name=logname.stem, samples=samples)
            except Exception as err:
                ConsoleOutput.print(f'Error while reading {logname}: {err}. It will be ignored by Shake&Tune!')
                continue

        return self.measurements

    def __del__(self):
        try:
            if self._temp_file.exists():
                self._temp_file.unlink()
        except Exception:
            pass  # Ignore errors during cleanup


class Accelerometer:
    def __init__(self, klipper_accelerometer, k_reactor):
        self._k_accelerometer = klipper_accelerometer
        self._k_reactor = k_reactor
        self._bg_client = None
        self._measurements_manager: MeasurementsManager = None
        self._samples_ready = False
        self._sample_error = None

    @staticmethod
    def find_axis_accelerometer(printer, axis: str = 'xy'):
        accel_chip_names = printer.lookup_object('resonance_tester').accel_chip_names
        for chip_axis, chip_name in accel_chip_names:
            if axis in {'x', 'y'} and chip_axis == 'xy':
                return chip_name
            elif chip_axis == axis:
                return chip_name
        return None

    def start_recording(self, measurements_manager: MeasurementsManager, name: str = None, append_time: bool = True):
        if self._bg_client is None:
            self._bg_client = self._k_accelerometer.start_internal_client()

            timestamp = time.strftime('%Y%m%d_%H%M%S')
            if name is None:
                name = timestamp
            elif append_time:
                name += f'_{timestamp}'

            if not name.replace('-', '').replace('_', '').isalnum():
                raise ValueError('Invalid measurement name!')

            self._measurements_manager = measurements_manager
            self._measurements_manager.add_measurement(name=name)
        else:
            raise ValueError('Recording already started!')

    def stop_recording(self) -> MeasurementsManager:
        if self._bg_client is None:
            ConsoleOutput.print('Warning: no recording to stop!')
            return None

        # Register a callback in Klipper's reactor to finish the measurements and get the
        # samples when Klipper is ready to process them (and without blocking its main thread)
        self._k_reactor.register_callback(self._finish_and_get_samples)
        self._wait_for_samples()

        return self._measurements_manager

    def _finish_and_get_samples(self, bg_client):
        try:
            self._bg_client.finish_measurements()
            samples = self._bg_client.samples or self._bg_client.get_samples()
            self._measurements_manager.append_samples_to_current_measurement(samples)
            self._samples_ready = True
        except Exception as e:
            ConsoleOutput.print(f'Error during accelerometer data retrieval: {e}')
            self._sample_error = e
        finally:
            self._bg_client = None

    def _wait_for_samples(self, timeout: int = WAIT_FOR_SAMPLE_TIMEOUT):
        eventtime = self._k_reactor.monotonic()
        endtime = eventtime + timeout

        while eventtime < endtime:
            if self._samples_ready:
                break
            if self._sample_error:
                raise self._sample_error
            eventtime = self._k_reactor.pause(eventtime + 0.05)

        if not self._samples_ready:
            raise TimeoutError(
                'Shake&Tune was unable to retrieve accelerometer data in time. '
                'This might be due to slow hardware or a busy system.'
            )

        self._samples_ready = False
