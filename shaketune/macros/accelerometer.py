#!/usr/bin/env python3

import time

from ..helpers.console_output import ConsoleOutput


class Accelerometer:
    def __init__(self, klipper_accelerometer):
        self._k_accelerometer = klipper_accelerometer

    def start_measurement(self):
        if self._k_accelerometer.bg_client is None:
            self._k_accelerometer.bg_client = self._k_accelerometer.chip.start_internal_client()
            ConsoleOutput.print('accelerometer measurements started')
        else:
            raise ValueError('measurements already started!')

    def stop_measurement(self, name: str = None):
        if self._k_accelerometer.bg_client is not None:
            name = name or time.strftime('%Y%m%d_%H%M%S')
            if not name.replace('-', '').replace('_', '').isalnum():
                raise ValueError('invalid file name!')

            bg_client = self._k_accelerometer.bg_client
            self._k_accelerometer.bg_client = None
            bg_client.finish_measurements()

            filename = f'/tmp/shaketune-{name}.csv'
            self._write_to_file(bg_client, filename)
            ConsoleOutput.print(f'Measurements stopped. Data written to {filename}')
        else:
            raise ValueError('measurements need to be started first!')

    def _write_to_file(self, bg_client, filename):
        with open(filename, 'w') as f:
            f.write('#time,accel_x,accel_y,accel_z\n')
            samples = bg_client.samples or bg_client.get_samples()
            for t, accel_x, accel_y, accel_z in samples:
                f.write('%.6f,%.6f,%.6f,%.6f\n' % (t, accel_x, accel_y, accel_z))
