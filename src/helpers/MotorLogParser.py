#!/usr/bin/env python3

# Classes to parse the Klipper log and parse the TMC dump to extract the relevant information
# Written by Frix_x#0161 #

import re
from pathlib import Path
from typing import Optional


class Motor:
    def __init__(self, name: str):
        self._name = name
        self._registers = {}
        self._properties = {}

    def set_register(self, register, value):
        self._registers[register] = value

    def set_property(self, property, value):
        self._properties[property] = value

    def get_register(self, register):
        return self._registers.get(register)

    def get_property(self, property):
        return self._properties.get(property)

    def __str__(self):
        return f'{self._name}\nProperties: {self._properties}\nRegisters: {self._registers}'


class MotorLogParser:
    _section_pattern = r'DUMP_TMC stepper_(x|y)'
    _register_patterns = {
        'CHOPCONF': r'CHOPCONF:\s+\S+\s+(.*)',
        'PWMCONF': r'PWMCONF:\s+\S+\s+(.*)',
        'COOLCONF': r'COOLCONF:\s+(.*)',
        'TPWMTHRS': r'TPWMTHRS:\s+\S+\s+(.*)',
        'TCOOLTHRS': r'TCOOLTHRS:\s+\S+\s+(.*)',
    }

    def __init__(self, filepath: Path, config_string: Optional[str] = None):
        self._filepath = filepath
        self._motors = {}
        self._config = self._parse_config(config_string) if config_string else {}
        self._parse_registers()

    def _parse_config(self, config_string: str):
        config = {}
        if config_string:
            entries = config_string.split('|')
            for entry in entries:
                if entry:
                    key, value = entry.split(':')
                    config[key.strip()] = self._convert_value(value.strip())
        return config

    def _convert_value(self, value: str):
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            return value

    def _parse_registers(self):
        with open(self._filepath, 'r') as file:
            log_content = file.read()

        sections = re.split(self._section_pattern, log_content)
        stepper_sections = {'stepper_x': None, 'stepper_y': None}

        for i in range(1, len(sections), 2):
            stepper = 'stepper_' + sections[i].strip()
            content = sections[i + 1]
            stepper_sections[stepper] = content

        for stepper, content in stepper_sections.items():
            if content:
                motor = Motor(stepper)
                # Apply the general properties from config string
                for key, value in self._config.items():
                    if stepper in key:
                        prop_key = key.replace(stepper + '_', '')
                        motor.set_property(prop_key, value)

                # Parse the specific registers
                for key, pattern in self._register_patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        values = match.group(1).strip()
                        if key == 'CHOPCONF':
                            mres_match = re.search(r'mres=\d+\((\d+)usteps\)', values)
                            if mres_match:
                                values = re.sub(r'mres=\d+\(\d+usteps\)', f'mres={mres_match.group(1)}', values)
                        cleaned_values = re.sub(r'\b\w+:\s+\S+\s+', '', values)
                        motor.set_register(key, cleaned_values)
                self._motors[stepper] = motor

    def get_motor(self, motor_name: str):
        return self._motors.get(motor_name)

    def get_motors(self):
        return self._motors


# # Usage example:
#     config_string = "tmc_x_name:tmc2240|x_run_current:0.9|x_hold_current:0.9|tmc_y_name:tmc2240|y_run_current:0.9|y_hold_current:0.9|autotune_enabled:True|x_motor:ldo-35sth48-1684ah|x_voltage:24.0|y_motor:ldo-35sth48-1684ah|y_voltage:24.0|"
#     parser = MotorLogParser('/path/to/your/logfile.log', config_string)

#     stepper_x = parser.get_motor('stepper_x')
#     stepper_y = parser.get_motor('stepper_y')

#     print(stepper_x)
#     print(stepper_y)
