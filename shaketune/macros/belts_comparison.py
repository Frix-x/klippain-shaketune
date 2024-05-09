#!/usr/bin/env python3


from ..helpers.console_output import ConsoleOutput
from ..shaketune_thread import ShakeTuneThread


def compare_belts_responses(gcmd, gcode, printer, st_thread: ShakeTuneThread) -> None:
    min_freq = gcmd.get_float('FREQ_START', default=5, minval=1)
    max_freq = gcmd.get_float('FREQ_END', default=133.33, minval=1)
    hz_per_sec = gcmd.get_float('HZ_PER_SEC', default=1, minval=1)

    toolhead = printer.lookup_object('toolhead')

    gcode.run_script_from_command(
        f'TEST_RESONANCES AXIS=1,1 OUTPUT=raw_data NAME=b FREQ_START={min_freq} FREQ_END={max_freq} HZ_PER_SEC={hz_per_sec}'
    )
    toolhead.wait_moves()

    gcode.run_script_from_command(
        f'TEST_RESONANCES AXIS=1,-1 OUTPUT=raw_data NAME=a FREQ_START={min_freq} FREQ_END={max_freq} HZ_PER_SEC={hz_per_sec}'
    )
    toolhead.wait_moves()

    # Run post-processing
    ConsoleOutput.print('Belts comparative frequency profile generation...')
    ConsoleOutput.print('This may take some time (3-5min)')
    st_thread.run()
