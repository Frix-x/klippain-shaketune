#!/usr/bin/env python3

from ..helpers.common_func import AXIS_CONFIG
from ..helpers.console_output import ConsoleOutput
from .resonance_test import vibrate_axis


def excitate_axis_at_freq(gcmd, config) -> None:
    freq = gcmd.get_int('FREQUENCY', default=25, minval=1)
    duration = gcmd.get_int('DURATION', default=10, minval=1)
    accel_per_hz = gcmd.get_float('ACCEL_PER_HZ', default=None)
    axis = gcmd.get('AXIS', default='x').lower()
    feedrate_travel = gcmd.get_float('TRAVEL_SPEED', default=120.0, minval=20.0)
    z_height = gcmd.get_float('Z_HEIGHT', default=None, minval=1)

    axis_config = next((item for item in AXIS_CONFIG if item['axis'] == axis), None)
    if axis_config is None:
        raise gcmd.error('AXIS selection invalid. Should be either x, y, a or b!')

    ConsoleOutput.print(f'Excitating {axis.upper()} axis at {freq}Hz for {duration} seconds')

    printer = config.get_printer()
    gcode = printer.lookup_object('gcode')
    toolhead = printer.lookup_object('toolhead')
    res_tester = printer.lookup_object('resonance_tester')
    systime = printer.get_reactor().monotonic()

    if accel_per_hz is None:
        accel_per_hz = res_tester.test.accel_per_hz

    # Move to the starting point
    test_points = res_tester.test.get_start_test_points()
    if len(test_points) > 1:
        raise gcmd.error('Only one test point in the [resonance_tester] section is supported by Shake&Tune.')
    if test_points[0] == (-1, -1, -1):
        if z_height is None:
            raise gcmd.error(
                'Z_HEIGHT parameter is required if the test_point in [resonance_tester] section is set to -1,-1,-1'
            )
        # Use center of bed in case the test point in [resonance_tester] is set to -1,-1,-1
        # This is usefull to get something automatic and is also used in the Klippain modular config
        kin_info = toolhead.kin.get_status(systime)
        mid_x = (kin_info['axis_minimum'].x + kin_info['axis_maximum'].x) / 2
        mid_y = (kin_info['axis_minimum'].y + kin_info['axis_maximum'].y) / 2
        point = (mid_x, mid_y, z_height)
    else:
        x, y, z = test_points[0]
        if z_height is not None:
            z = z_height
        point = (x, y, z)

    toolhead.manual_move(point, feedrate_travel)
    toolhead.dwell(0.5)

    min_freq = freq - 1
    max_freq = freq + 1
    hz_per_sec = 1 / (duration / 3)
    vibrate_axis(toolhead, gcode, axis_config['direction'], min_freq, max_freq, hz_per_sec, accel_per_hz)
