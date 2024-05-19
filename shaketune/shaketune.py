#!/usr/bin/env python3


import os
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
        self._pconfig = config
        self._printer = config.get_printer()
        gcode = self._printer.lookup_object('gcode')

        res_tester = self._printer.lookup_object('resonance_tester')
        if res_tester is None:
            config.error('No [resonance_tester] config section found in printer.cfg! Please add one to use Shake&Tune.')

        self.timeout = config.getfloat('timeout', 2.0, above=0.0)
        result_folder = config.get('result_folder', default='~/printer_data/config/ShakeTune_results')
        result_folder_path = Path(result_folder).expanduser() if result_folder else None
        keep_n_results = config.getint('number_of_results_to_keep', default=3, minval=0)
        keep_csv = config.getboolean('keep_raw_csv', default=False)
        show_macros = config.getboolean('show_macros_in_webui', default=True)
        dpi = config.getint('dpi', default=150, minval=100, maxval=500)

        self._config = ShakeTuneConfig(result_folder_path, keep_n_results, keep_csv, dpi)
        ConsoleOutput.register_output_callback(gcode.respond_info)

        commands = [
            ('EXCITATE_AXIS_AT_FREQ', self.cmd_EXCITATE_AXIS_AT_FREQ, 'Maintain a specified excitation frequency for a period of time to diagnose and locate a source of vibration'),
            ('AXES_MAP_CALIBRATION', self.cmd_AXES_MAP_CALIBRATION, 'Perform a set of movements to measure the orientation of the accelerometer and help you set the best axes_map configuration for your printer'),
            ('COMPARE_BELTS_RESPONSES', self.cmd_COMPARE_BELTS_RESPONSES, 'Perform a custom half-axis test to analyze and compare the frequency profiles of individual belts on CoreXY printers'),
            ('AXES_SHAPER_CALIBRATION', self.cmd_AXES_SHAPER_CALIBRATION, 'Perform standard axis input shaper tests on one or both XY axes to select the best input shaper filter'),
            ('CREATE_VIBRATIONS_PROFILE', self.cmd_CREATE_VIBRATIONS_PROFILE, 'Perform a set of movements to measure the orientation of the accelerometer and help you set the best axes_map configuration for your printer')
        ]
        command_descriptions = {name: desc for name, _, desc in commands}

        for name, command, description in commands:
            gcode.register_command(
                f'_{name}' if show_macros else name,
                command,
                desc=description
            )

        # Load the dummy macros with their description in order to show them in the web interfaces
        if show_macros:
            pconfig = self._printer.lookup_object('configfile')
            dirname = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(dirname, 'dummy_macros.cfg')
            try:
                dummy_macros = pconfig.read_config(filename)
            except Exception as err:
                raise config.error("Cannot load Shake&Tune dummy macro '%s'" % (filename,)) from err
            for macro in dummy_macros.get_prefix_sections(''):
                section = macro.get_name()
                command = section.split(' ', 1)[1]
                if command in command_descriptions:
                    description = command_descriptions[command]
                else:
                    description = 'Shake&Tune macro'
                macro.fileconfig._sections[section]['description'] = description
                self._printer.load_object(dummy_macros, section)


    def cmd_EXCITATE_AXIS_AT_FREQ(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        excitate_axis_at_freq(gcmd, self._pconfig)

    def cmd_AXES_MAP_CALIBRATION(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        axes_map_finder = AxesMapFinder(self._config)
        st_thread = ShakeTuneThread(self._config, axes_map_finder, self._printer.get_reactor(), self.timeout)
        axes_map_calibration(gcmd, self._pconfig, st_thread)

    def cmd_COMPARE_BELTS_RESPONSES(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        belt_graph_creator = BeltsGraphCreator(self._config)
        st_thread = ShakeTuneThread(self._config, belt_graph_creator, self._printer.get_reactor(), self.timeout)
        compare_belts_responses(gcmd, self._pconfig, st_thread)

    def cmd_AXES_SHAPER_CALIBRATION(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        shaper_graph_creator = ShaperGraphCreator(self._config)
        st_thread = ShakeTuneThread(self._config, shaper_graph_creator, self._printer.get_reactor(), self.timeout)
        axes_shaper_calibration(gcmd, self._pconfig, st_thread)

    def cmd_CREATE_VIBRATIONS_PROFILE(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        vibration_profile_creator = VibrationsGraphCreator(self._config)
        st_thread = ShakeTuneThread(self._config, vibration_profile_creator, self._printer.get_reactor(), self.timeout)
        create_vibrations_profile(gcmd, self._pconfig, st_thread)
