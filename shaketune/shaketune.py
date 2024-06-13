# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: shaketune.py
# Description: Main class implementation for Shake&Tune, handling Klipper initialization and
#              loading of the plugin, and the registration of the tuning commands


import os
from pathlib import Path

from .commands import (
    axes_map_calibration,
    axes_shaper_calibration,
    compare_belts_responses,
    create_vibrations_profile,
    excitate_axis_at_freq,
)
from .graph_creators import (
    AxesMapGraphCreator,
    BeltsGraphCreator,
    ShaperGraphCreator,
    StaticGraphCreator,
    VibrationsGraphCreator,
)
from .helpers.console_output import ConsoleOutput
from .shaketune_config import ShakeTuneConfig
from .shaketune_process import ShakeTuneProcess


class ShakeTune:
    def __init__(self, config) -> None:
        self._pconfig = config
        self._printer = config.get_printer()
        gcode = self._printer.lookup_object('gcode')

        res_tester = self._printer.lookup_object('resonance_tester', None)
        if res_tester is None:
            config.error('No [resonance_tester] config section found in printer.cfg! Please add one to use Shake&Tune.')

        self.timeout = config.getfloat('timeout', 300, above=0.0)
        result_folder = config.get('result_folder', default='~/printer_data/config/ShakeTune_results')
        result_folder_path = Path(result_folder).expanduser() if result_folder else None
        keep_n_results = config.getint('number_of_results_to_keep', default=3, minval=0)
        keep_csv = config.getboolean('keep_raw_csv', default=False)
        show_macros = config.getboolean('show_macros_in_webui', default=True)
        dpi = config.getint('dpi', default=150, minval=100, maxval=500)

        self._config = ShakeTuneConfig(result_folder_path, keep_n_results, keep_csv, dpi)
        ConsoleOutput.register_output_callback(gcode.respond_info)

        commands = [
            (
                'EXCITATE_AXIS_AT_FREQ',
                self.cmd_EXCITATE_AXIS_AT_FREQ,
                'Maintain a specified excitation frequency for a period of time to diagnose and locate a source of vibration',
            ),
            (
                'AXES_MAP_CALIBRATION',
                self.cmd_AXES_MAP_CALIBRATION,
                'Perform a set of movements to measure the orientation of the accelerometer and help you set the best axes_map configuration for your printer',
            ),
            (
                'COMPARE_BELTS_RESPONSES',
                self.cmd_COMPARE_BELTS_RESPONSES,
                'Perform a custom half-axis test to analyze and compare the frequency profiles of individual belts on CoreXY printers',
            ),
            (
                'AXES_SHAPER_CALIBRATION',
                self.cmd_AXES_SHAPER_CALIBRATION,
                'Perform standard axis input shaper tests on one or both XY axes to select the best input shaper filter',
            ),
            (
                'CREATE_VIBRATIONS_PROFILE',
                self.cmd_CREATE_VIBRATIONS_PROFILE,
                'Perform a set of movements to measure the orientation of the accelerometer and help you set the best axes_map configuration for your printer',
            ),
        ]
        command_descriptions = {name: desc for name, _, desc in commands}

        for name, command, description in commands:
            gcode.register_command(f'_{name}' if show_macros else name, command, desc=description)

        # Load the dummy macros with their description in order to show them in the web interfaces
        if show_macros:
            pconfig = self._printer.lookup_object('configfile')
            dirname = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(dirname, 'dummy_macros.cfg')
            try:
                dummy_macros_cfg = pconfig.read_config(filename)
            except Exception as err:
                raise config.error(f'Cannot load Shake&Tune dummy macro {filename}') from err

            for gcode_macro in dummy_macros_cfg.get_prefix_sections('gcode_macro '):
                gcode_macro_name = gcode_macro.get_name()

                # Replace the dummy description by the one here (to avoid code duplication and define it in only one place)
                command = gcode_macro_name.split(' ', 1)[1]
                description = command_descriptions.get(command, 'Shake&Tune macro')
                gcode_macro.fileconfig.set(gcode_macro_name, 'description', description)

                # Add the section to the Klipper configuration object with all its options
                if not config.fileconfig.has_section(gcode_macro_name.lower()):
                    config.fileconfig.add_section(gcode_macro_name.lower())
                for option in gcode_macro.fileconfig.options(gcode_macro_name):
                    value = gcode_macro.fileconfig.get(gcode_macro_name, option)
                    config.fileconfig.set(gcode_macro_name.lower(), option, value)

                    # Small trick to ensure the new injected sections are considered valid by Klipper config system
                    config.access_tracking[(gcode_macro_name.lower(), option.lower())] = 1

                # Finally, load the section within the printer objects
                self._printer.load_object(config, gcode_macro_name.lower())

    def cmd_EXCITATE_AXIS_AT_FREQ(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        static_freq_graph_creator = StaticGraphCreator(self._config)
        st_process = ShakeTuneProcess(self._config, static_freq_graph_creator, self.timeout)
        excitate_axis_at_freq(gcmd, self._pconfig, st_process)

    def cmd_AXES_MAP_CALIBRATION(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        axes_map_graph_creator = AxesMapGraphCreator(self._config)
        st_process = ShakeTuneProcess(self._config, axes_map_graph_creator, self.timeout)
        axes_map_calibration(gcmd, self._pconfig, st_process)

    def cmd_COMPARE_BELTS_RESPONSES(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        belt_graph_creator = BeltsGraphCreator(self._config)
        st_process = ShakeTuneProcess(self._config, belt_graph_creator, self.timeout)
        compare_belts_responses(gcmd, self._pconfig, st_process)

    def cmd_AXES_SHAPER_CALIBRATION(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        shaper_graph_creator = ShaperGraphCreator(self._config)
        st_process = ShakeTuneProcess(self._config, shaper_graph_creator, self.timeout)
        axes_shaper_calibration(gcmd, self._pconfig, st_process)

    def cmd_CREATE_VIBRATIONS_PROFILE(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        vibration_profile_creator = VibrationsGraphCreator(self._config)
        st_process = ShakeTuneProcess(self._config, vibration_profile_creator, self.timeout)
        create_vibrations_profile(gcmd, self._pconfig, st_process)
