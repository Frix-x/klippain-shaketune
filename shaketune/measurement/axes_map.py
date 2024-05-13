#!/usr/bin/env python3


from ..helpers.console_output import ConsoleOutput
from ..shaketune_thread import ShakeTuneThread
from .accelerometer import Accelerometer


def axes_map_calibration(gcmd, gcode, printer, st_thread: ShakeTuneThread) -> None:
    z_height = gcmd.get_float('Z_HEIGHT', default=20.0)
    speed = gcmd.get_float('SPEED', default=80.0, minval=20.0)
    accel = gcmd.get_int('ACCEL', default=1500, minval=100)
    feedrate_travel = gcmd.get_float('TRAVEL_SPEED', default=120.0, minval=20.0)
    accel_chip = gcmd.get('ACCEL_CHIP', default=None)

    if accel_chip is None:
        accel_chip = Accelerometer.find_axis_accelerometer(printer, 'xy')
        if accel_chip is None:
            gcmd.error(
                'No accelerometer specified for measurement! Multi-accelerometer configurations are not supported for this macro.'
            )
    accelerometer = Accelerometer(printer.lookup_object(accel_chip))

    systime = printer.get_reactor().monotonic()
    toolhead = printer.lookup_object('toolhead')
    toolhead_info = toolhead.get_status(systime)
    old_accel = toolhead_info['max_accel']
    old_mcr = toolhead_info['minimum_cruise_ratio']
    old_sqv = toolhead_info['square_corner_velocity']

    # set the wanted acceleration values
    gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={accel} MINIMUM_CRUISE_RATIO=0 SQUARE_CORNER_VELOCITY=5.0')

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
    toolhead.move([mid_x - 15, mid_y - 15, z_height, E], feedrate_travel)
    toolhead.dwell(0.5)

    # Start the measurements and do the movements (+X, +Y and then +Z)
    accelerometer.start_measurement()
    toolhead.dwell(1)
    toolhead.move([mid_x + 15, mid_y - 15, z_height, E], speed)
    toolhead.dwell(1)
    toolhead.move([mid_x + 15, mid_y + 15, z_height, E], speed)
    toolhead.dwell(1)
    toolhead.move([mid_x + 15, mid_y + 15, z_height + 15, E], speed)
    toolhead.dwell(1)
    accelerometer.stop_measurement('axemap')

    # Re-enable the input shaper if it was active
    if input_shaper is not None:
        input_shaper.enable_shaping()

    # Restore the previous acceleration values
    gcode.run_script_from_command(
        f'SET_VELOCITY_LIMIT ACCEL={old_accel} MINIMUM_CRUISE_RATIO={old_mcr} SQUARE_CORNER_VELOCITY={old_sqv}'
    )
    toolhead.wait_moves()

    # Run post-processing
    ConsoleOutput.print('Analysis of the movements...')
    creator = st_thread.get_graph_creator()
    creator.configure(accel)
    st_thread.run()
