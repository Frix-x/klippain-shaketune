# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: accelerometer.py
# Description: Provides a custom and internal Shake&Tune Accelerometer helper that interfaces
#              with Klipper's accelerometer classes. It includes functions to start and stop
#              accelerometer measurements and write the data to a file in a blocking manner.


import os
import time
from multiprocessing import Process, Queue

FILE_WRITE_TIMEOUT = 10  # seconds


class Accelerometer:
    def __init__(self, reactor, klipper_accelerometer):
        self._k_accelerometer = klipper_accelerometer
        self._reactor = reactor

        self._bg_client = None
        self._write_queue = Queue()
        self._write_processes = []

    @staticmethod
    def find_axis_accelerometer(printer, axis: str = 'xy'):
        accel_chip_names = printer.lookup_object('resonance_tester').accel_chip_names
        for chip_axis, chip_name in accel_chip_names:
            if axis in {'x', 'y'} and chip_axis == 'xy':
                return chip_name
            elif chip_axis == axis:
                return chip_name
        return None

    def start_measurement(self):
        if self._bg_client is None:
            self._bg_client = self._k_accelerometer.start_internal_client()
        else:
            raise ValueError('measurements already started!')

    def stop_measurement(self, name: str = None, append_time: bool = True):
        if self._bg_client is None:
            raise ValueError('measurements need to be started first!')

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if name is None:
            name = timestamp
        elif append_time:
            name += f'_{timestamp}'

        if not name.replace('-', '').replace('_', '').isalnum():
            raise ValueError('invalid file name!')

        bg_client = self._bg_client
        self._bg_client = None
        bg_client.finish_measurements()

        filename = f'/tmp/shaketune-{name}.csv'
        self._queue_file_write(bg_client, filename)

    def _queue_file_write(self, bg_client, filename):
        self._write_queue.put(filename)
        write_proc = Process(target=self._write_to_file, args=(bg_client, filename))
        write_proc.daemon = True
        write_proc.start()
        self._write_processes.append(write_proc)

    def _write_to_file(self, bg_client, filename):
        try:
            os.nice(20)
        except Exception:
            pass

        with open(filename, 'w') as f:
            f.write('#time,accel_x,accel_y,accel_z\n')
            samples = bg_client.samples or bg_client.get_samples()
            for t, accel_x, accel_y, accel_z in samples:
                f.write(f'{t:.6f},{accel_x:.6f},{accel_y:.6f},{accel_z:.6f}\n')

        self._write_queue.get()

    def wait_for_file_writes(self):
        while not self._write_queue.empty():
            eventtime = self._reactor.monotonic()
            self._reactor.pause(eventtime + 0.1)

        for proc in self._write_processes:
            if proc is None:
                continue
            eventtime = self._reactor.monotonic()
            endtime = eventtime + FILE_WRITE_TIMEOUT
            complete = False
            while eventtime < endtime:
                eventtime = self._reactor.pause(eventtime + 0.05)
                if not proc.is_alive():
                    complete = True
                    break
            if not complete:
                raise TimeoutError(
                    'Shake&Tune was not able to write the accelerometer data into the CSV file. '
                    'This might be due to a slow SD card or a busy or full filesystem.'
                )

        self._write_processes = []
