# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: axes_shaper_calibration.py
# Description: Provides a command for calibrating the input shaper of a 3D printer's axes using an accelerometer.
#              The script performs resonance tests along specified axes, starts and stops measurements,
#              and generates graphs for each axis to analyze the collected data.


from ..helpers.accelerometer import Accelerometer, MeasurementsManager
from ..helpers.common_func import AXIS_CONFIG
from ..helpers.console_output import ConsoleOutput
from ..helpers.resonance_test import vibrate_axis
from ..shaketune_process import ShakeTuneProcess


def axes_shaper_calibration(gcmd, config, st_process: ShakeTuneProcess) -> None:
    printer = config.get_printer()
    toolhead = printer.lookup_object('toolhead')
    res_tester = printer.lookup_object('resonance_tester')
    systime = printer.get_reactor().monotonic()
    toolhead_info = toolhead.get_status(systime)

    # Get the default values for the frequency range and the acceleration per Hz
    if hasattr(res_tester, 'test'):
        # Old Klipper code (before Dec 6, 2024: https://github.com/Klipper3d/klipper/commit/16b4b6b302ac3ffcd55006cd76265aad4e26ecc8)
        default_min_freq = res_tester.test.min_freq
        default_max_freq = res_tester.test.max_freq
        default_accel_per_hz = res_tester.test.accel_per_hz
        test_points = res_tester.test.get_start_test_points()
    else:
        # New Klipper code (after Dec 6, 2024) with the sweeping test
        default_min_freq = res_tester.generator.vibration_generator.min_freq
        default_max_freq = res_tester.generator.vibration_generator.max_freq
        default_accel_per_hz = res_tester.generator.vibration_generator.accel_per_hz
        test_points = res_tester.probe_points

    min_freq = gcmd.get_float('FREQ_START', default=default_min_freq, minval=1)
    max_freq = gcmd.get_float('FREQ_END', default=default_max_freq, minval=1)
    hz_per_sec = gcmd.get_float('HZ_PER_SEC', default=1, minval=1)
    accel_per_hz = gcmd.get_float('ACCEL_PER_HZ', default=None)
    axis_input = gcmd.get('AXIS', default='all').lower()
    if axis_input not in {'x', 'y', 'all'}:
        raise gcmd.error('AXIS selection invalid. Should be either x, y, or all!')
    scv = gcmd.get_float('SCV', default=toolhead_info['square_corner_velocity'], minval=0)
    max_sm = gcmd.get_float('MAX_SMOOTHING', default=None, minval=0)
    feedrate_travel = gcmd.get_float('TRAVEL_SPEED', default=120.0, minval=20.0)
    z_height = gcmd.get_float('Z_HEIGHT', default=None, minval=1)

    if accel_per_hz == '':
        accel_per_hz = None

    if accel_per_hz is None:
        accel_per_hz = default_accel_per_hz

    gcode = printer.lookup_object('gcode')

    max_accel = max_freq * accel_per_hz

    # Move to the starting point
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
    if 'minimum_cruise_ratio' in toolhead_info:  # minimum_cruise_ratio found: Klipper >= v0.12.0-239
        old_mcr = toolhead_info['minimum_cruise_ratio']
        gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={max_accel} MINIMUM_CRUISE_RATIO=0')
    else:  # minimum_cruise_ratio not found: Klipper < v0.12.0-239
        old_mcr = None
        gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={max_accel}')

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
        measurements_manager = MeasurementsManager(st_process.get_st_config().chunk_size)

        # First we need to find the accelerometer chip suited for the axis
        accel_chip = Accelerometer.find_axis_accelerometer(printer, config['axis'])
        if accel_chip is None:
            raise gcmd.error('No suitable accelerometer found for measurement!')
        accelerometer = Accelerometer(printer.lookup_object(accel_chip), printer.get_reactor())

        # Then do the actual measurements
        ConsoleOutput.print(f'Measuring {config["label"]}...')
        accelerometer.start_recording(measurements_manager, name=config['label'], append_time=True)
        vibrate_axis(toolhead, gcode, config['direction'], min_freq, max_freq, hz_per_sec, accel_per_hz, res_tester)
        accelerometer.stop_recording()
        accelerometer.wait_for_samples()
        toolhead.dwell(0.5)
        toolhead.wait_moves()

        # And finally generate the graph for each measured axis
        ConsoleOutput.print(f'{config["axis"].upper()} axis frequency profile generation...')
        ConsoleOutput.print('This may take some time (1-3min)')
        measurements_manager.wait_for_data_transfers(printer.get_reactor())
        st_process.run(measurements_manager)
        st_process.wait_for_completion()
        toolhead.dwell(1)

    # Re-enable the input shaper if it was active
    if input_shaper is not None:
        input_shaper.enable_shaping()

    # Restore the previous acceleration values
    if old_mcr is not None:  # minimum_cruise_ratio found: Klipper >= v0.12.0-239
        gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_accel} MINIMUM_CRUISE_RATIO={old_mcr}')
    else:  # minimum_cruise_ratio not found: Klipper < v0.12.0-239
        gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_accel}')
