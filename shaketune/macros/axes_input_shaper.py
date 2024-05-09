#!/usr/bin/env python3


from ..helpers.console_output import ConsoleOutput
from ..shaketune_thread import ShakeTuneThread


def axes_shaper_calibration(gcmd, gcode, printer, st_thread: ShakeTuneThread) -> None:
    min_freq = gcmd.get_float('FREQ_START', default=5, minval=1)
    max_freq = gcmd.get_float('FREQ_END', default=133.33, minval=1)
    hz_per_sec = gcmd.get_float('HZ_PER_SEC', default=1, minval=1)
    axis = gcmd.get('AXIS', default='all')
    if axis not in ['x', 'y', 'all']:
        gcmd.error('AXIS selection invalid. Should be either x, y, or all!')
    scv = gcmd.get_float('SCV', default=None, minval=0)
    max_sm = gcmd.get_float('MAX_SMOOTHING', default=None, minval=0)

    if scv is None:
        systime = printer.get_reactor().monotonic()
        toolhead = printer.lookup_object('toolhead')
        toolhead_info = toolhead.get_status(systime)
        scv = toolhead_info['square_corner_velocity']

    creator = st_thread.get_graph_creator()
    creator.configure(scv, max_sm)

    axis_flags = {'x': axis in ('x', 'all'), 'y': axis in ('y', 'all')}
    for axis in ['x', 'y']:
        if axis_flags[axis]:
            gcode.run_script_from_command(
                f'TEST_RESONANCES AXIS={axis.upper()} OUTPUT=raw_data NAME={axis} FREQ_START={min_freq} FREQ_END={max_freq} HZ_PER_SEC={hz_per_sec}'
            )
            ConsoleOutput.print(f'{axis.upper()} axis frequency profile generation...')
            ConsoleOutput.print('This may take some time (1-3min)')
            st_thread.run()
