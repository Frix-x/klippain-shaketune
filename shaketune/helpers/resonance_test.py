# Shake&Tune: 3D printer analysis tools
#
# Adapted from Klipper's original resonance_tester.py file by Dmitry Butyugin <dmbutyugin@google.com>
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: resonance_test.py
# Description: Contains functions to test the resonance frequency of the printer and its components
#              by vibrating the toolhead in specific axis directions. This derive a bit from Klipper's
#              implementation as there are two main changes:
#                1. Original code doesn't use euclidean distance with projection for the coordinates calculation.
#                   The new approach implemented here ensures that the vector's total length remains constant (= L),
#                   regardless of the direction components. It's especially important when the direction vector
#                   involves combinations of movements along multiple axes like for the diagonal belt tests.
#                2. Original code doesn't allow Z axis movements that was added in order to test the Z axis resonance
#                   or CoreXZ belts frequency profiles as well.


import math

from ..helpers.console_output import ConsoleOutput


# This function is used to vibrate the toolhead in a specific axis direction
# to test the resonance frequency of the printer and its components
def vibrate_axis(toolhead, gcode, axis_direction, min_freq, max_freq, hz_per_sec, accel_per_hz):
    freq = min_freq
    X, Y, Z, E = toolhead.get_position()
    sign = 1.0

    while freq <= max_freq + 0.000001:
        t_seg = 0.25 / freq  # Time segment for one vibration cycle
        accel = accel_per_hz * freq  # Acceleration for each half-cycle
        max_v = accel * t_seg  # Max velocity for each half-cycle
        toolhead.cmd_M204(gcode.create_gcode_command('M204', 'M204', {'S': accel}))
        L = 0.5 * accel * t_seg**2  # Distance for each half-cycle

        # Calculate move points based on axis direction (X, Y and Z)
        magnitude = math.sqrt(sum([component**2 for component in axis_direction]))
        normalized_direction = tuple(component / magnitude for component in axis_direction)
        dX, dY, dZ = normalized_direction[0] * L, normalized_direction[1] * L, normalized_direction[2] * L
        nX = X + sign * dX
        nY = Y + sign * dY
        nZ = Z + sign * dZ

        # Execute movement
        toolhead.move([nX, nY, nZ, E], max_v)
        toolhead.move([X, Y, Z, E], max_v)
        sign *= -1

        # Increase frequency for next cycle
        old_freq = freq
        freq += 2 * t_seg * hz_per_sec
        if int(freq) > int(old_freq):
            ConsoleOutput.print(f'Testing frequency: {freq:.0f} Hz')

    toolhead.wait_moves()


# This function is used to vibrate the toolhead in a specific axis direction at a static frequency for a specific duration
def vibrate_axis_at_static_freq(toolhead, gcode, axis_direction, freq, duration, accel_per_hz):
    X, Y, Z, E = toolhead.get_position()
    sign = 1.0

    # Compute movements values
    t_seg = 0.25 / freq
    accel = accel_per_hz * freq
    max_v = accel * t_seg
    toolhead.cmd_M204(gcode.create_gcode_command('M204', 'M204', {'S': accel}))
    L = 0.5 * accel * t_seg**2

    # Calculate move points based on axis direction (X, Y and Z)
    magnitude = math.sqrt(sum([component**2 for component in axis_direction]))
    normalized_direction = tuple(component / magnitude for component in axis_direction)
    dX, dY, dZ = normalized_direction[0] * L, normalized_direction[1] * L, normalized_direction[2] * L

    # Start a timer to measure the duration of the test and execute the vibration within the specified time
    start_time = toolhead.reactor.monotonic()
    while toolhead.reactor.monotonic() - start_time < duration:
        nX = X + sign * dX
        nY = Y + sign * dY
        nZ = Z + sign * dZ
        toolhead.move([nX, nY, nZ, E], max_v)
        toolhead.move([X, Y, Z, E], max_v)
        sign *= -1

    toolhead.wait_moves()
