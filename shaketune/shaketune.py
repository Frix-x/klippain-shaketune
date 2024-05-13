#!/usr/bin/env python3


from pathlib import Path

from .helpers.console_output import ConsoleOutput
from .measurement import (
    axes_map_calibration,
    axes_shaper_calibration,
    compare_belts_responses,
    create_vibrations_profile,
    excitate_axis_at_freq,
)
from .post_processing import AxesMapFinder, BeltsGraphCreator, ShaperGraphCreator, VibrationsGraphCreator
from .shaketune_config import ShakeTuneConfig
from .shaketune_thread import ShakeTuneThread


class ShakeTune:
    def __init__(self, config) -> None:
        self._printer = config.get_printer()
        self._gcode = self._printer.lookup_object('gcode')

        res_tester = self._printer.lookup_object('resonance_tester')
        if res_tester is None:
            config.error('No [resonance_tester] config section found in printer.cfg! Please add one to use Shake&Tune')

        self.timeout = config.getfloat('timeout', 2.0, above=0.0)

        result_folder = config.get('result_folder', default='~/printer_data/config/K-ShakeTune_results')
        result_folder_path = Path(result_folder).expanduser() if result_folder else None
        keep_n_results = config.getint('number_of_results_to_keep', default=3, minval=0)
        keep_csv = config.getboolean('keep_raw_csv', default=False)
        dpi = config.getint('dpi', default=150, minval=100, maxval=500)

        self._config = ShakeTuneConfig(result_folder_path, keep_n_results, keep_csv, dpi)
        ConsoleOutput.register_output_callback(self._gcode.respond_info)

        self._gcode.register_command(
            'EXCITATE_AXIS_AT_FREQ',
            self.cmd_EXCITATE_AXIS_AT_FREQ,
            desc=self.cmd_EXCITATE_AXIS_AT_FREQ_help,
        )
        self._gcode.register_command(
            'AXES_MAP_CALIBRATION',
            self.cmd_AXES_MAP_CALIBRATION,
            desc=self.cmd_AXES_MAP_CALIBRATION_help,
        )
        self._gcode.register_command(
            'COMPARE_BELTS_RESPONSES',
            self.cmd_COMPARE_BELTS_RESPONSES,
            desc=self.cmd_COMPARE_BELTS_RESPONSES_help,
        )
        self._gcode.register_command(
            'AXES_SHAPER_CALIBRATION',
            self.cmd_AXES_SHAPER_CALIBRATION,
            desc=self.cmd_AXES_SHAPER_CALIBRATION_help,
        )
        self._gcode.register_command(
            'CREATE_VIBRATIONS_PROFILE',
            self.cmd_CREATE_VIBRATIONS_PROFILE,
            desc=self.cmd_CREATE_VIBRATIONS_PROFILE_help,
        )

    cmd_EXCITATE_AXIS_AT_FREQ_help = (
        'Maintain a specified excitation frequency for a period of time to diagnose and locate a source of vibration'
    )

    def cmd_EXCITATE_AXIS_AT_FREQ(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        excitate_axis_at_freq(gcmd, self._gcode, self._printer)

    cmd_AXES_MAP_CALIBRATION_help = 'Perform a set of movements to measure the orientation of the accelerometer and help you set the best axes_map configuration for your printer'

    def cmd_AXES_MAP_CALIBRATION(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        axes_map_finder = AxesMapFinder(self._config)
        st_thread = ShakeTuneThread(self._config, axes_map_finder, self._printer.get_reactor(), self.timeout)
        axes_map_calibration(gcmd, self._gcode, self._printer, st_thread)

    cmd_COMPARE_BELTS_RESPONSES_help = 'Perform a custom half-axis test to analyze and compare the frequency profiles of individual belts on CoreXY printers'

    def cmd_COMPARE_BELTS_RESPONSES(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        belt_graph_creator = BeltsGraphCreator(self._config)
        st_thread = ShakeTuneThread(self._config, belt_graph_creator, self._printer.get_reactor(), self.timeout)
        compare_belts_responses(gcmd, self._gcode, self._printer, st_thread)

    cmd_AXES_SHAPER_CALIBRATION_help = (
        'Perform standard axis input shaper tests on one or both XY axes to select the best input shaper filter'
    )

    def cmd_AXES_SHAPER_CALIBRATION(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        shaper_graph_creator = ShaperGraphCreator(self._config)
        st_thread = ShakeTuneThread(self._config, shaper_graph_creator, self._printer.get_reactor(), self.timeout)
        axes_shaper_calibration(gcmd, self._gcode, self._printer, st_thread)

    cmd_CREATE_VIBRATIONS_PROFILE_help = 'Perform a set of movements to measure the orientation of the accelerometer and help you set the best axes_map configuration for your printer'

    def cmd_CREATE_VIBRATIONS_PROFILE(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        vibration_profile_creator = VibrationsGraphCreator(self._config)
        st_thread = ShakeTuneThread(self._config, vibration_profile_creator, self._printer.get_reactor(), self.timeout)
        create_vibrations_profile(gcmd, self._gcode, self._printer, st_thread)
