#!/usr/bin/env python3

# This file provides a custom and internal Shake&Tune Accelerometer helper that is
# an interface to Klipper's own accelerometer classes. It is used to start and
# stop accelerometer measurements and write the data to a file in a blocking manner.

import time

# from ..helpers.console_output import ConsoleOutput


class Accelerometer:
    def __init__(self, klipper_accelerometer):
        self._k_accelerometer = klipper_accelerometer
        self._bg_client = None

    @staticmethod
    def find_axis_accelerometer(printer, axis: str = 'xy'):
        accel_chip_names = printer.lookup_object('resonance_tester').accel_chip_names
        for chip_axis, chip_name in accel_chip_names:
            if axis in ['x', 'y'] and chip_axis == 'xy':
                return chip_name
            elif chip_axis == axis:
                return chip_name
        return None

    def start_measurement(self):
        if self._bg_client is None:
            self._bg_client = self._k_accelerometer.start_internal_client()
            # ConsoleOutput.print('Accelerometer measurements started')
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
        self._write_to_file(bg_client, filename)
        # ConsoleOutput.print(f'Accelerometer measurements stopped. Data written to {filename}')

    def _write_to_file(self, bg_client, filename):
        with open(filename, 'w') as f:
            f.write('#time,accel_x,accel_y,accel_z\n')
            samples = bg_client.samples or bg_client.get_samples()
            for t, accel_x, accel_y, accel_z in samples:
                f.write('%.6f,%.6f,%.6f,%.6f\n' % (t, accel_x, accel_y, accel_z))
