# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: axes_map_calibration.py
# Description: Provides a command for calibrating the axes map of a 3D printer using an accelerometer.
#              The script moves the printer head along specified axes, starts and stops measurements,
#              and performs post-processing to analyze the collected data.

from datetime import datetime

from ..helpers.accelerometer import Accelerometer, MeasurementsManager
from ..helpers.console_output import ConsoleOutput
from ..shaketune_process import ShakeTuneProcess

SEGMENT_LENGTH = 30  # mm


def axes_map_calibration(gcmd, config, st_process: ShakeTuneProcess) -> None:
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    z_height = gcmd.get_float('Z_HEIGHT', default=20.0)
    speed = gcmd.get_float('SPEED', default=80.0, minval=20.0)
    accel = gcmd.get_int('ACCEL', default=1500, minval=100)
    feedrate_travel = gcmd.get_float('TRAVEL_SPEED', default=120.0, minval=20.0)

    printer = config.get_printer()
    gcode = printer.lookup_object('gcode')
    toolhead = printer.lookup_object('toolhead')
    systime = printer.get_reactor().monotonic()

    accel_chip = Accelerometer.find_axis_accelerometer(printer, 'xy')
    k_accelerometer = printer.lookup_object(accel_chip, None)
    if k_accelerometer is None:
        raise gcmd.error('Multi-accelerometer configurations are not supported for this macro!')
    pconfig = printer.lookup_object('configfile')
    current_axes_map = pconfig.status_raw_config[accel_chip].get('axes_map', None)
    accelerometer = Accelerometer(k_accelerometer, printer.get_reactor())

    toolhead_info = toolhead.get_status(systime)
    old_accel = toolhead_info['max_accel']
    old_sqv = toolhead_info['square_corner_velocity']

    # set the wanted acceleration values
    if 'minimum_cruise_ratio' in toolhead_info:
        old_mcr = toolhead_info['minimum_cruise_ratio']  # minimum_cruise_ratio found: Klipper >= v0.12.0-239
        gcode.run_script_from_command(
            f'SET_VELOCITY_LIMIT ACCEL={accel} MINIMUM_CRUISE_RATIO=0 SQUARE_CORNER_VELOCITY=5.0'
        )
    else:  # minimum_cruise_ratio not found: Klipper < v0.12.0-239
        old_mcr = None
        gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={accel} SQUARE_CORNER_VELOCITY=5.0')

    # Deactivate input shaper if it is active to get raw movements
    input_shaper = printer.lookup_object('input_shaper', None)
    if input_shaper is not None:
        input_shaper.disable_shaping()
    else:
        input_shaper = None

    kin_info = toolhead.kin.get_status(systime)
    mid_x = (kin_info['axis_minimum'].x + kin_info['axis_maximum'].x) / 2
    mid_y = (kin_info['axis_minimum'].y + kin_info['axis_maximum'].y) / 2
    _, _, _, E = toolhead.get_position()

    # Going to the start position
    toolhead.move([mid_x - SEGMENT_LENGTH / 2, mid_y - SEGMENT_LENGTH / 2, z_height, E], feedrate_travel)
    toolhead.dwell(0.5)

    creator = st_process.get_graph_creator()
    filename = creator.get_folder() / f'{creator.get_type().replace(" ", "")}_{date}'
    measurements_manager = MeasurementsManager(st_process.get_st_config().chunk_size, printer.get_reactor(), filename)

    # Start the measurements and do the movements (+X, +Y and then +Z)
    accelerometer.start_recording(measurements_manager, name='axesmap_X', append_time=True)
    toolhead.dwell(0.5)
    toolhead.move([mid_x + SEGMENT_LENGTH / 2, mid_y - SEGMENT_LENGTH / 2, z_height, E], speed)
    toolhead.dwell(0.5)
    accelerometer.stop_recording()
    toolhead.dwell(0.5)
    accelerometer.start_recording(measurements_manager, name='axesmap_Y', append_time=True)
    toolhead.dwell(0.5)
    toolhead.move([mid_x + SEGMENT_LENGTH / 2, mid_y + SEGMENT_LENGTH / 2, z_height, E], speed)
    toolhead.dwell(0.5)
    accelerometer.stop_recording()
    toolhead.dwell(0.5)
    accelerometer.start_recording(measurements_manager, name='axesmap_Z', append_time=True)
    toolhead.dwell(0.5)
    toolhead.move([mid_x + SEGMENT_LENGTH / 2, mid_y + SEGMENT_LENGTH / 2, z_height + SEGMENT_LENGTH, E], speed)
    toolhead.dwell(0.5)
    accelerometer.stop_recording()
    toolhead.dwell(0.5)

    # Re-enable the input shaper if it was active
    if input_shaper is not None:
        input_shaper.enable_shaping()

    # Restore the previous acceleration values
    if old_mcr is not None:  # minimum_cruise_ratio found: Klipper >= v0.12.0-239
        gcode.run_script_from_command(
            f'SET_VELOCITY_LIMIT ACCEL={old_accel} MINIMUM_CRUISE_RATIO={old_mcr} SQUARE_CORNER_VELOCITY={old_sqv}'
        )
    else:  # minimum_cruise_ratio not found: Klipper < v0.12.0-239
        gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_accel} SQUARE_CORNER_VELOCITY={old_sqv}')

    toolhead.wait_moves()

    # Run post-processing
    ConsoleOutput.print('Analysis of the movements...')
    ConsoleOutput.print('This may take some time (1-3min)')
    creator.configure(accel, current_axes_map)
    creator.define_output_target(filename)
    measurements_manager.save_stdata()
    st_process.run(filename)
    st_process.wait_for_completion()
