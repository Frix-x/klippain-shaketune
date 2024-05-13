#!/usr/bin/env python3


import math

from ..helpers.console_output import ConsoleOutput
from ..shaketune_thread import ShakeTuneThread
from .accelerometer import Accelerometer
from .motorsconfigparser import MotorsConfigParser

MIN_SPEED = 2  # mm/s


def create_vibrations_profile(gcmd, gcode, printer, st_thread: ShakeTuneThread) -> None:
    size = gcmd.get_float('SIZE', default=100.0, minval=50.0)
    z_height = gcmd.get_float('Z_HEIGHT', default=20.0)
    max_speed = gcmd.get_float('MAX_SPEED', default=200.0, minval=10.0)
    speed_increment = gcmd.get_float('SPEED_INCREMENT', default=2.0, minval=1.0)
    accel = gcmd.get_int('ACCEL', default=3000, minval=100)
    feedrate_travel = gcmd.get_float('TRAVEL_SPEED', default=120.0, minval=20.0)
    accel_chip = gcmd.get('ACCEL_CHIP', default=None)

    if (size / (max_speed / 60)) < 0.25:
        gcmd.error('The size of the movement is too small for the given speed! Increase SIZE or decrease MAX_SPEED!')

    # Check that input shaper is already configured
    input_shaper = printer.lookup_object('input_shaper', None)
    if input_shaper is None:
        gcmd.error('Input shaper is not configured! Please run the shaper calibration macro first.')

    # TODO: Add the kinematics check to define the main_angles
    #       but this needs to retrieve it from the printer configuration
    # {% if kinematics == "cartesian" %}
    #     # Cartesian motors are on X and Y axis directly
    #     RESPOND MSG="Cartesian kinematics mode"
    #     {% set main_angles = [0, 90] %}
    # {% elif kinematics == "corexy" %}
    #     # CoreXY motors are on A and B axis (45 and 135 degrees)
    #     RESPOND MSG="CoreXY kinematics mode"
    #     {% set main_angles = [45, 135] %}
    # {% else %}
    #     { action_raise_error("Only Cartesian and CoreXY kinematics are supported at the moment for the vibrations measurement tool!") }
    # {% endif %}
    kinematics = 'cartesian'
    main_angles = [0, 90]

    systime = printer.get_reactor().monotonic()
    toolhead = printer.lookup_object('toolhead')
    toolhead_info = toolhead.get_status(systime)
    old_accel = toolhead_info['max_accel']
    old_mcr = toolhead_info['minimum_cruise_ratio']
    old_sqv = toolhead_info['square_corner_velocity']

    # set the wanted acceleration values
    gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={accel} MINIMUM_CRUISE_RATIO=0 SQUARE_CORNER_VELOCITY=5.0')

    kin_info = toolhead.kin.get_status(systime)
    mid_x = (kin_info['axis_minimum'].x + kin_info['axis_maximum'].x) / 2
    mid_y = (kin_info['axis_minimum'].y + kin_info['axis_maximum'].y) / 2
    X, Y, _, E = toolhead.get_position()

    # Going to the start position
    toolhead.move([X, Y, z_height, E], feedrate_travel / 10)
    toolhead.move([mid_x - 15, mid_y - 15, z_height, E], feedrate_travel)
    toolhead.dwell(0.5)

    nb_speed_samples = int((max_speed - MIN_SPEED) / speed_increment + 1)
    for curr_angle in main_angles:
        radian_angle = math.radians(curr_angle)

        # Find the best accelerometer chip for the current angle if not specified
        if curr_angle == 0:
            accel_axis = 'x'
        elif curr_angle == 90:
            accel_axis = 'y'
        else:
            accel_axis = 'xy'
        if accel_chip is None:
            accel_chip = Accelerometer.find_axis_accelerometer(printer, accel_axis)
            if accel_chip is None:
                gcmd.error(
                    'No accelerometer specified for measurement! Multi-accelerometer configurations are not supported for this macro.'
                )
        accelerometer = Accelerometer(printer.lookup_object(accel_chip))

        # Sweep the speed range to record the vibrations at different speeds
        for curr_speed_sample in range(nb_speed_samples):
            curr_speed = MIN_SPEED + curr_speed_sample * speed_increment

            # Reduce the segments length for the lower speed range (0-100mm/s). The minimum length is 1/3 of the SIZE and is gradually increased
            # to the nominal SIZE at 100mm/s. No further size changes are made above this speed. The goal is to ensure that the print head moves
            # enough to collect enough data for vibration analysis, without doing unnecessary distance to save time. At higher speeds, the full
            # segments lengths are used because the head moves faster and travels more distance in the same amount of time and we want enough data
            if curr_speed < 100:
                segment_length_multiplier = 1 / 5 + 4 / 5 * curr_speed / 100
            else:
                segment_length_multiplier = 1

            # Calculate angle coordinates using trigonometry and length multiplier and move to start point
            dX = (size / 2) * math.cos(radian_angle) * segment_length_multiplier
            dY = (size / 2) * math.sin(radian_angle) * segment_length_multiplier
            toolhead.move([mid_x - dX, mid_y - dY, z_height, E], feedrate_travel)

            # Adjust the number of back and forth movements based on speed to also save time on lower speed range
            # 3 movements are done by default, reduced to 2 between 150-250mm/s and to 1 under 150mm/s.
            movements = 3
            if curr_speed < 150:
                movements = 1
            elif curr_speed < 250:
                movements = 2

            # Back and forth movements to record the vibrations at constant speed in both direction
            accelerometer.start_measurement()
            for _ in range(movements):
                toolhead.move([mid_x + dX, mid_y + dY, z_height, E], curr_speed)
                toolhead.move([mid_x - dX, mid_y - dY, z_height, E], curr_speed)
            name = f'vib_an{curr_angle:.2f}sp{curr_speed:.2f}'.replace('.', '_')
            accelerometer.stop_measurement(name)

            toolhead.dwell(0.3)
            toolhead.wait_moves()

    # Restore the previous acceleration values
    gcode.run_script_from_command(
        f'SET_VELOCITY_LIMIT ACCEL={old_accel} MINIMUM_CRUISE_RATIO={old_mcr} SQUARE_CORNER_VELOCITY={old_sqv}'
    )
    toolhead.wait_moves()

    # Get the motors and TMC configurations from Klipper
    motors_config_parser = MotorsConfigParser(printer, motors=['stepper_x', 'stepper_y'])

    # Run post-processing
    ConsoleOutput.print('Machine vibrations profile generation...')
    ConsoleOutput.print('This may take some time (5-8min)')
    creator = st_thread.get_graph_creator()
    creator.configure(kinematics, accel, motors_config_parser)
    st_thread.run()
