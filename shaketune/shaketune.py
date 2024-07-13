# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: shaketune.py
# Description: Main class implementation for Shake&Tune, handling Klipper initialization and
#              loading of the plugin, and the registration of the tuning commands


import importlib
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

DEFAULT_FOLDER = '~/printer_data/config/ShakeTune_results'
DEFAULT_NUMBER_OF_RESULTS = 3
DEFAULT_KEEP_RAW_CSV = False
DEFAULT_DPI = 150
DEFAULT_TIMEOUT = 300
DEFAULT_SHOW_MACROS = True
ST_COMMANDS = {
    'EXCITATE_AXIS_AT_FREQ': (
        'Maintain a specified excitation frequency for a period '
        'of time to diagnose and locate a source of vibrations'
    ),
    'AXES_MAP_CALIBRATION': (
        'Perform a set of movements to measure the orientation of the accelerometer '
        'and help you set the best axes_map configuration for your printer'
    ),
    'COMPARE_BELTS_RESPONSES': (
        'Perform a custom half-axis test to analyze and compare the '
        'frequency profiles of individual belts on CoreXY or CoreXZ printers'
    ),
    'AXES_SHAPER_CALIBRATION': 'Perform standard axis input shaper tests on one or both XY axes to select the best input shaper filter',
    'CREATE_VIBRATIONS_PROFILE': (
        'Run a series of motions to find speed/angle ranges where the printer could be '
        'exposed to VFAs to optimize your slicer speed profiles and TMC driver parameters'
    ),
}


class ShakeTune:
    def __init__(self, config) -> None:
        self._config = config
        self._printer = config.get_printer()
        self._printer.register_event_handler('klippy:connect', self._on_klippy_connect)

        # Check if Shake&Tune is running in DangerKlipper
        self.IN_DANGER = importlib.util.find_spec('extras.danger_options') is not None

        # Register the console print output callback to the corresponding Klipper function
        gcode = self._printer.lookup_object('gcode')
        ConsoleOutput.register_output_callback(gcode.respond_info)

        self._initialize_config(config)
        self._register_commands()

    # Initialize the ShakeTune object and its configuration
    def _initialize_config(self, config) -> None:
        result_folder = config.get('result_folder', default=DEFAULT_FOLDER)
        result_folder_path = Path(result_folder).expanduser() if result_folder else None
        keep_n_results = config.getint('number_of_results_to_keep', default=DEFAULT_NUMBER_OF_RESULTS, minval=0)
        keep_csv = config.getboolean('keep_raw_csv', default=DEFAULT_KEEP_RAW_CSV)
        dpi = config.getint('dpi', default=DEFAULT_DPI, minval=100, maxval=500)
        self._st_config = ShakeTuneConfig(result_folder_path, keep_n_results, keep_csv, dpi)

        self.timeout = config.getfloat('timeout', 300, above=0.0)
        self._show_macros = config.getboolean('show_macros_in_webui', default=True)

    # Create the Klipper commands to allow the user to run Shake&Tune's tools
    def _register_commands(self) -> None:
        gcode = self._printer.lookup_object('gcode')
        measurement_commands = [
            ('EXCITATE_AXIS_AT_FREQ', self.cmd_EXCITATE_AXIS_AT_FREQ, ST_COMMANDS['EXCITATE_AXIS_AT_FREQ']),
            ('AXES_MAP_CALIBRATION', self.cmd_AXES_MAP_CALIBRATION, ST_COMMANDS['AXES_MAP_CALIBRATION']),
            ('COMPARE_BELTS_RESPONSES', self.cmd_COMPARE_BELTS_RESPONSES, ST_COMMANDS['COMPARE_BELTS_RESPONSES']),
            ('AXES_SHAPER_CALIBRATION', self.cmd_AXES_SHAPER_CALIBRATION, ST_COMMANDS['AXES_SHAPER_CALIBRATION']),
            ('CREATE_VIBRATIONS_PROFILE', self.cmd_CREATE_VIBRATIONS_PROFILE, ST_COMMANDS['CREATE_VIBRATIONS_PROFILE']),
        ]

        # Register Shake&Tune's measurement commands using the official Klipper API (gcode.register_command)
        # Doing this makes the commands available in Klipper but they are not shown in the web interfaces
        # and are only available by typing the full name in the console (like all the other Klipper commands)
        for name, command, description in measurement_commands:
            gcode.register_command(f'_{name}' if self._show_macros else name, command, desc=description)

        # Then, a hack to inject the macros into Klipper's config system in order to show them in the web
        # interfaces. This is not a good way to do it, but it's the only way to do it for now to get
        # a good user experience while using Shake&Tune (it's indeed easier to just click a macro button)
        if self._show_macros:
            configfile = self._printer.lookup_object('configfile')
            dirname = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(dirname, 'dummy_macros.cfg')
            try:
                dummy_macros_cfg = configfile.read_config(filename)
            except Exception as err:
                raise self._config.error(f'Cannot load Shake&Tune dummy macro {filename}') from err

            for gcode_macro in dummy_macros_cfg.get_prefix_sections('gcode_macro '):
                gcode_macro_name = gcode_macro.get_name()

                # Replace the dummy description by the one from ST_COMMANDS (to avoid code duplication and define it in only one place)
                command = gcode_macro_name.split(' ', 1)[1]
                description = ST_COMMANDS.get(command, 'Shake&Tune macro')
                gcode_macro.fileconfig.set(gcode_macro_name, 'description', description)

                # Add the section to the Klipper configuration object with all its options
                if not self._config.fileconfig.has_section(gcode_macro_name.lower()):
                    self._config.fileconfig.add_section(gcode_macro_name.lower())
                for option in gcode_macro.fileconfig.options(gcode_macro_name):
                    value = gcode_macro.fileconfig.get(gcode_macro_name, option)
                    self._config.fileconfig.set(gcode_macro_name.lower(), option, value)
                    # Small trick to ensure the new injected sections are considered valid by Klipper config system
                    self._config.access_tracking[(gcode_macro_name.lower(), option.lower())] = 1

                # Finally, load the section within the printer objects
                self._printer.load_object(self._config, gcode_macro_name.lower())

    def _on_klippy_connect(self) -> None:
        # Check if the resonance_tester object is available in the printer
        # configuration as it is required for Shake&Tune to work properly
        res_tester = self._printer.lookup_object('resonance_tester', None)
        if res_tester is None:
            raise self._config.error(
                'No [resonance_tester] config section found in printer.cfg! Please add one to use Shake&Tune!'
            )

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # Following are all the Shake&Tune commands that are registered to the Klipper console
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------

    def cmd_EXCITATE_AXIS_AT_FREQ(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        static_freq_graph_creator = StaticGraphCreator(self._st_config)
        st_process = ShakeTuneProcess(
            self._st_config,
            self._printer.get_reactor(),
            static_freq_graph_creator,
            self.timeout,
        )
        excitate_axis_at_freq(gcmd, self._config, st_process)

    def cmd_AXES_MAP_CALIBRATION(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        axes_map_graph_creator = AxesMapGraphCreator(self._st_config)
        st_process = ShakeTuneProcess(
            self._st_config,
            self._printer.get_reactor(),
            axes_map_graph_creator,
            self.timeout,
        )
        axes_map_calibration(gcmd, self._config, st_process)

    def cmd_COMPARE_BELTS_RESPONSES(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        belt_graph_creator = BeltsGraphCreator(self._st_config)
        st_process = ShakeTuneProcess(
            self._st_config,
            self._printer.get_reactor(),
            belt_graph_creator,
            self.timeout,
        )
        compare_belts_responses(gcmd, self._config, st_process)

    def cmd_AXES_SHAPER_CALIBRATION(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        shaper_graph_creator = ShaperGraphCreator(self._st_config)
        st_process = ShakeTuneProcess(
            self._st_config,
            self._printer.get_reactor(),
            shaper_graph_creator,
            self.timeout,
        )
        axes_shaper_calibration(gcmd, self._config, st_process)

    def cmd_CREATE_VIBRATIONS_PROFILE(self, gcmd) -> None:
        ConsoleOutput.print(f'Shake&Tune version: {ShakeTuneConfig.get_git_version()}')
        vibration_profile_creator = VibrationsGraphCreator(self._st_config)
        st_process = ShakeTuneProcess(
            self._st_config,
            self._printer.get_reactor(),
            vibration_profile_creator,
            self.timeout,
        )
        create_vibrations_profile(gcmd, self._config, st_process)
