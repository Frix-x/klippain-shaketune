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


import os
import pickle
import time
import uuid
from multiprocessing import Process
from pathlib import Path
from typing import List, Tuple, TypedDict

import numpy as np
import zstandard as zstd

from ..helpers.console_output import ConsoleOutput

Sample = Tuple[float, float, float, float]
SamplesList = List[Sample]

CHUNK_SIZE = 15  # Maximum number of measurements to keep in memory at once


class Measurement(TypedDict):
    name: str
    samples: SamplesList


class MeasurementsManager:
    def __init__(self, chunk_size: int):
        self._chunk_size = chunk_size
        self.measurements: List[Measurement] = []
        self._uuid = str(uuid.uuid4())[:8]
        self._temp_dir = Path(f'/tmp/shaketune_{self._uuid}')
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_files = []
        self._write_processes = []

    def clear_measurements(self, keep_last: bool = False):
        self.measurements = [self.measurements[-1]] if keep_last else []

    def append_samples_to_last_measurement(self, additional_samples: SamplesList):
        try:
            self.measurements[-1]['samples'].extend(additional_samples)
        except IndexError as err:
            raise ValueError('no measurements available to append samples to.') from err

    def add_measurement(self, name: str, samples: SamplesList = None):
        samples = samples if samples is not None else []
        self.measurements.append({'name': name, 'samples': samples})
        if len(self.measurements) > self._chunk_size:
            self._save_chunk()

    def _save_chunk(self):
        # Save the measurements to the chunk file. We keep the last measurement
        # in memory to be able to append new samples to it later if needed
        chunk_filename = self._temp_dir / f'{self._uuid}_{len(self._chunk_files)}.stchunk'
        process = Process(target=self._save_to_file, args=(chunk_filename, self.measurements[:-1].copy()))
        process.daemon = False
        process.start()
        self._write_processes.append(process)
        self._chunk_files.append(chunk_filename)
        self.clear_measurements(keep_last=True)

    def save_stdata(self, filename: Path):
        process = Process(target=self._reassemble_chunks, args=(filename,))
        process.daemon = False
        process.start()
        self._write_processes.append(process)

    def _reassemble_chunks(self, filename: Path):
        try:
            os.nice(19)
        except Exception:
            pass  # Ignore errors as it's not critical
        try:
            all_measurements = []
            for chunk_file in self._chunk_files:
                chunk_measurements = self._load_measurements_from_file(chunk_file)
                all_measurements.extend(chunk_measurements)
                os.remove(chunk_file)  # Remove the chunk file after reading

            # Include any remaining measurements in memory
            if self.measurements:
                all_measurements.extend(self.measurements)

            # Save all measurements to the final .stdata file
            self._save_to_file(filename, all_measurements)

            # Clean up
            self.clear_measurements()
            self._chunk_files = []
        except Exception as e:
            ConsoleOutput.print(f'Warning: unable to assemble chunks into {filename}: {e}')

    def _save_to_file(self, filename: Path, measurements: List[Measurement]):
        try:
            os.nice(19)
        except Exception:
            pass  # Ignore errors as it's not critical
        try:
            with open(filename, 'wb') as f:
                cctx = zstd.ZstdCompressor(level=3)
                with cctx.stream_writer(f) as compressor:
                    pickle.dump(measurements, compressor)
        except Exception as e:
            ConsoleOutput.print(f'Warning: unable to save the data to {filename}: {e}')

    def wait_for_data_transfers(self, k_reactor, timeout: int = 30):
        if not self._write_processes:
            return  # No file write is pending

        eventtime = k_reactor.monotonic()
        endtime = eventtime + timeout
        complete = False

        while eventtime < endtime:
            eventtime = k_reactor.pause(eventtime + 0.05)
            if all(not p.is_alive() for p in self._write_processes):
                complete = True
                break

        if not complete:
            raise TimeoutError(
                'Shake&Tune was unable to write the accelerometer data on the filesystem. '
                'This might be due to a slow, busy or full SD card.'
            )

        self._write_processes = []

    def _load_measurements_from_file(self, filename: Path) -> List[Measurement]:
        try:
            with open(filename, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as decompressor:
                    measurements = pickle.load(decompressor)
            return measurements
        except Exception as e:
            ConsoleOutput.print(f'Warning: unable to load measurements from {filename}: {e}')
            return []

    def get_measurements(self) -> List[Measurement]:
        all_measurements = []
        for chunk_file in self._chunk_files:
            chunk_measurements = self._load_measurements_from_file(chunk_file)
            all_measurements.extend(chunk_measurements)
        all_measurements.extend(self.measurements)  # Include any remaining measurements in memory

        return all_measurements

    def load_from_stdata(self, filename: Path) -> List[Measurement]:
        self.measurements = self._load_measurements_from_file(filename)
        return self.measurements

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
            if self._temp_dir.exists():
                for chunk_file in self._temp_dir.glob('*.stchunk'):
                    chunk_file.unlink()
                self._temp_dir.rmdir()
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
                raise ValueError('invalid measurement name!')

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

        return self._measurements_manager

    def _finish_and_get_samples(self, bg_client):
        try:
            self._bg_client.finish_measurements()
            samples = self._bg_client.samples or self._bg_client.get_samples()
            self._measurements_manager.append_samples_to_last_measurement(samples)
            self._samples_ready = True
        except Exception as e:
            ConsoleOutput.print(f'Error during accelerometer data retrieval: {e}')
            self._sample_error = e
        finally:
            self._bg_client = None

    def wait_for_samples(self, timeout: int = 60):
        eventtime = self._k_reactor.monotonic()
        endtime = eventtime + timeout

        while eventtime < endtime:
            eventtime = self._k_reactor.pause(eventtime + 0.05)
            if self._samples_ready:
                break
            if self._sample_error:
                raise self._sample_error

        if not self._samples_ready:
            raise TimeoutError(
                'Shake&Tune was unable to retrieve accelerometer data in time. '
                'This might be due to slow hardware or a busy system.'
            )

        self._samples_ready = False
