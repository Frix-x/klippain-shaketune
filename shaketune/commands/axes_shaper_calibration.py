#!/usr/bin/env python3


from ..helpers.common_func import AXIS_CONFIG
from ..helpers.console_output import ConsoleOutput
from ..helpers.resonance_test import vibrate_axis
from ..shaketune_process import ShakeTuneProcess
from .accelerometer import Accelerometer


def axes_shaper_calibration(gcmd, config, st_process: ShakeTuneProcess) -> None:
    min_freq = gcmd.get_float('FREQ_START', default=5, minval=1)
    max_freq = gcmd.get_float('FREQ_END', default=133.33, minval=1)
    hz_per_sec = gcmd.get_float('HZ_PER_SEC', default=1, minval=1)
    accel_per_hz = gcmd.get_float('ACCEL_PER_HZ', default=None)
    axis_input = gcmd.get('AXIS', default='all').lower()
    if axis_input not in ['x', 'y', 'all']:
        raise gcmd.error('AXIS selection invalid. Should be either x, y, or all!')
    scv = gcmd.get_float('SCV', default=None, minval=0)
    max_sm = gcmd.get_float('MAX_SMOOTHING', default=None, minval=0)
    feedrate_travel = gcmd.get_float('TRAVEL_SPEED', default=120.0, minval=20.0)
    z_height = gcmd.get_float('Z_HEIGHT', default=None, minval=1)

    if accel_per_hz == '':
        accel_per_hz = None

    printer = config.get_printer()
    gcode = printer.lookup_object('gcode')
    toolhead = printer.lookup_object('toolhead')
    res_tester = printer.lookup_object('resonance_tester')
    systime = printer.get_reactor().monotonic()

    if scv is None:
        toolhead_info = toolhead.get_status(systime)
        scv = toolhead_info['square_corner_velocity']

    if accel_per_hz is None:
        accel_per_hz = res_tester.test.accel_per_hz
    max_accel = max_freq * accel_per_hz

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

    # Configure the graph creator
    creator = st_process.get_graph_creator()
    creator.configure(scv, max_sm, accel_per_hz)

    # set the needed acceleration values for the test
    toolhead_info = toolhead.get_status(systime)
    old_accel = toolhead_info['max_accel']
    old_mcr = toolhead_info['minimum_cruise_ratio']
    gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={max_accel} MINIMUM_CRUISE_RATIO=0')

    # Deactivate input shaper if it is active to get raw movements
    input_shaper = printer.lookup_object('input_shaper', None)
    if input_shaper is not None:
        input_shaper.disable_shaping()
    else:
        input_shaper = None

    # Filter axis configurations based on user input, assuming 'axis_input' can be 'x', 'y', 'all' (that means 'x' and 'y')
    filtered_config = [
        a for a in AXIS_CONFIG if a['axis'] == axis_input or (axis_input == 'all' and a['axis'] in ('x', 'y'))
    ]
    for config in filtered_config:
        # First we need to find the accelerometer chip suited for the axis
        accel_chip = Accelerometer.find_axis_accelerometer(printer, config['axis'])
        if accel_chip is None:
            raise gcmd.error('No suitable accelerometer found for measurement!')
        accelerometer = Accelerometer(printer.lookup_object(accel_chip))

        # Then do the actual measurements
        accelerometer.start_measurement()
        vibrate_axis(toolhead, gcode, config['direction'], min_freq, max_freq, hz_per_sec, accel_per_hz)
        accelerometer.stop_measurement(config['label'], append_time=True)

        # And finally generate the graph for each measured axis
        ConsoleOutput.print(f'{config["axis"].upper()} axis frequency profile generation...')
        ConsoleOutput.print('This may take some time (1-3min)')
        st_process.run()
        st_process.wait_for_completion()
        toolhead.dwell(1)
        toolhead.wait_moves()

    # Re-enable the input shaper if it was active
    if input_shaper is not None:
        input_shaper.enable_shaping()

    # Restore the previous acceleration values
    gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_accel} MINIMUM_CRUISE_RATIO={old_mcr}')
