# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: compare_belts_responses.py
# Description: Provides a command for comparing the frequency response of belts in CoreXY and CoreXZ kinematics 3D printers.
#              The script performs resonance tests along specified axes, starts and stops measurements, and generates graphs
#              for each axis to analyze the collected data.


from ..helpers.common_func import AXIS_CONFIG
from ..helpers.console_output import ConsoleOutput
from ..helpers.motors_config_parser import MotorsConfigParser
from ..helpers.resonance_test import vibrate_axis
from ..shaketune_process import ShakeTuneProcess
from .accelerometer import Accelerometer


def compare_belts_responses(gcmd, config, st_process: ShakeTuneProcess) -> None:
    printer = config.get_printer()
    toolhead = printer.lookup_object('toolhead')
    res_tester = printer.lookup_object('resonance_tester')
    systime = printer.get_reactor().monotonic()

    min_freq = gcmd.get_float('FREQ_START', default=res_tester.test.min_freq, minval=1)
    max_freq = gcmd.get_float('FREQ_END', default=res_tester.test.max_freq, minval=1)
    hz_per_sec = gcmd.get_float('HZ_PER_SEC', default=1, minval=1)
    accel_per_hz = gcmd.get_float('ACCEL_PER_HZ', default=None)
    feedrate_travel = gcmd.get_float('TRAVEL_SPEED', default=120.0, minval=20.0)
    z_height = gcmd.get_float('Z_HEIGHT', default=None, minval=1)

    if accel_per_hz == '':
        accel_per_hz = None

    if accel_per_hz is None:
        accel_per_hz = res_tester.test.accel_per_hz

    gcode = printer.lookup_object('gcode')

    max_accel = max_freq * accel_per_hz

    # Configure the graph creator
    motors_config_parser = MotorsConfigParser(config, motors=None)
    creator = st_process.get_graph_creator()
    creator.configure(motors_config_parser.kinematics, accel_per_hz)

    if motors_config_parser.kinematics == 'corexy':
        filtered_config = [a for a in AXIS_CONFIG if a['axis'] in ('a', 'b')]
        accel_chip = Accelerometer.find_axis_accelerometer(printer, 'xy')
    elif motors_config_parser.kinematics == 'corexz':
        filtered_config = [a for a in AXIS_CONFIG if a['axis'] in ('corexz_x', 'corexz_z')]
        # For CoreXZ kinematics, we can use the X axis accelerometer as most of the time they are moving bed printers
        accel_chip = Accelerometer.find_axis_accelerometer(printer, 'x')
    else:
        raise gcmd.error('Only CoreXY and CoreXZ kinematics are supported for the belt comparison tool!')
    ConsoleOutput.print(f'{motors_config_parser.kinematics.upper()} kinematics mode')

    if accel_chip is None:
        raise gcmd.error(
            'No suitable accelerometer found for measurement! Multi-accelerometer configurations are not supported for this macro.'
        )
    accelerometer = Accelerometer(printer.lookup_object(accel_chip))

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

    # Run the test for each axis
    for config in filtered_config:
        accelerometer.start_measurement()
        vibrate_axis(toolhead, gcode, config['direction'], min_freq, max_freq, hz_per_sec, accel_per_hz)
        accelerometer.stop_measurement(config['label'], append_time=True)

    accelerometer.wait_for_file_writes()

    # Re-enable the input shaper if it was active
    if input_shaper is not None:
        input_shaper.enable_shaping()

    # Restore the previous acceleration values
    gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_accel} MINIMUM_CRUISE_RATIO={old_mcr}')

    # Run post-processing
    ConsoleOutput.print('Belts comparative frequency profile generation...')
    ConsoleOutput.print('This may take some time (1-3min)')
    st_process.run()
    st_process.wait_for_completion()
