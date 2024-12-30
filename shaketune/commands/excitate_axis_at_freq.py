# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: excitate_axis_at_freq.py
# Description: Provide a command to excites a specified axis at a given frequency for a duration
#              and optionally creates a graph of the vibration data collected by the accelerometer.


from ..helpers.accelerometer import Accelerometer, MeasurementsManager
from ..helpers.common_func import AXIS_CONFIG
from ..helpers.compat import res_tester_config
from ..helpers.console_output import ConsoleOutput
from ..helpers.resonance_test import vibrate_axis_at_static_freq
from ..shaketune_process import ShakeTuneProcess


def excitate_axis_at_freq(gcmd, config, st_process: ShakeTuneProcess) -> None:
    create_graph = gcmd.get_int('CREATE_GRAPH', default=0, minval=0, maxval=1) == 1
    freq = gcmd.get_int('FREQUENCY', default=25, minval=1)
    duration = gcmd.get_int('DURATION', default=30, minval=1)
    accel_per_hz = gcmd.get_float('ACCEL_PER_HZ', default=None)
    axis = gcmd.get('AXIS', default='x').lower()
    feedrate_travel = gcmd.get_float('TRAVEL_SPEED', default=120.0, minval=20.0)
    z_height = gcmd.get_float('Z_HEIGHT', default=None, minval=1)
    accel_chip = gcmd.get('ACCEL_CHIP', default=None)

    if accel_chip == '':
        accel_chip = None
    if accel_per_hz == '':
        accel_per_hz = None

    axis_config = next((item for item in AXIS_CONFIG if item['axis'] == axis), None)
    if axis_config is None:
        raise gcmd.error('AXIS selection invalid. Should be either x, y, a or b!')

    if create_graph:
        printer = config.get_printer()
        if accel_chip is None:
            accel_chip = Accelerometer.find_axis_accelerometer(printer, 'xy' if axis in {'a', 'b'} else axis)
        k_accelerometer = printer.lookup_object(accel_chip, None)
        if k_accelerometer is None:
            raise gcmd.error(f'Accelerometer chip [{accel_chip}] was not found!')
        accelerometer = Accelerometer(k_accelerometer, printer.get_reactor())
        measurements_manager = MeasurementsManager(st_process.get_st_config().chunk_size)

    ConsoleOutput.print(f'Excitating {axis.upper()} axis at {freq}Hz for {duration} seconds')

    printer = config.get_printer()
    gcode = printer.lookup_object('gcode')
    toolhead = printer.lookup_object('toolhead')
    systime = printer.get_reactor().monotonic()

    # Get the default values for the acceleration per Hz and the test points
    default_min_freq, default_max_freq, default_accel_per_hz, test_points = res_tester_config(config)

    if accel_per_hz is None:
        accel_per_hz = default_accel_per_hz

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

    # Deactivate input shaper if it is active to get raw movements
    input_shaper = printer.lookup_object('input_shaper', None)
    if input_shaper is not None:
        input_shaper.disable_shaping()
    else:
        input_shaper = None

    # If the user want to create a graph, we start accelerometer recording
    if create_graph:
        accelerometer.start_recording(measurements_manager, name=f'staticfreq_{axis.upper()}', append_time=True)

    toolhead.dwell(0.5)
    vibrate_axis_at_static_freq(toolhead, gcode, axis_config['direction'], freq, duration, accel_per_hz)
    toolhead.dwell(0.5)

    # Re-enable the input shaper if it was active
    if input_shaper is not None:
        input_shaper.enable_shaping()

    # If the user wanted to create a graph, we stop the recording and generate it
    if create_graph:
        accelerometer.stop_recording()
        accelerometer.wait_for_samples()
        toolhead.dwell(0.5)

        creator = st_process.get_graph_creator()
        creator.configure(freq, duration, accel_per_hz)
        measurements_manager.wait_for_data_transfers(printer.get_reactor())
        st_process.run(measurements_manager)
        st_process.wait_for_completion()
