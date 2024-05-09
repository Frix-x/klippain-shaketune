#!/usr/bin/env python3

from ..helpers.console_output import ConsoleOutput


def excitate_axis_at_freq(gcmd, gcode) -> None:
    freq = gcmd.get_int('FREQUENCY', default=25, minval=1)
    duration = gcmd.get_int('DURATION', default=10, minval=1)
    axis = gcmd.get('AXIS', default='x')
    if axis not in ['x', 'y', 'a', 'b']:
        gcmd.error('AXIS selection invalid. Should be either x, y, a or b!')

    ConsoleOutput.print(f'Excitating {axis.upper()} axis at {freq}Hz for {duration} seconds')

    if axis == 'a':
        axis = '1,-1'
    elif axis == 'b':
        axis = '1,1'

    gcode.run_script_from_command(
        f'TEST_RESONANCES OUTPUT=raw_data AXIS={axis} FREQ_START={freq-1} FREQ_END={freq+1} HZ_PER_SEC={1/(duration/3)}'
    )
