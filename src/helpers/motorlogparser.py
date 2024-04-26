#!/usr/bin/env python3

# Classes to parse the Klipper log and parse the TMC dump to extract the relevant information
# Written by Frix_x#0161 #

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class Motor:
    def __init__(self, name: str):
        self._name: str = name
        self._registers: Dict[str, Any] = {}
        self._properties: Dict[str, Any] = {}

    def set_register(self, register: str, value: Any) -> None:
        self._registers[register] = value

    def set_property(self, property: str, value: Any) -> None:
        self._properties[property] = value

    def get_register(self, register: str) -> Optional[Any]:
        return self._registers.get(register)

    def get_property(self, property: str) -> Optional[Any]:
        return self._properties.get(property)

    def __str__(self):
        return f'Stepper: {self._name}\nKlipper config: {self._properties}\nTMC Registers: {self._registers}'


class MotorLogParser:
    _section_pattern: str = r'DUMP_TMC stepper_(x|y)'
    _register_patterns: Dict[str, str] = {
        'CHOPCONF': r'CHOPCONF:\s+\S+\s+(.*)',
        'PWMCONF': r'PWMCONF:\s+\S+\s+(.*)',
        'COOLCONF': r'COOLCONF:\s+(.*)',
        'TPWMTHRS': r'TPWMTHRS:\s+\S+\s+(.*)',
        'TCOOLTHRS': r'TCOOLTHRS:\s+\S+\s+(.*)',
    }

    def __init__(self, filepath: Path, config_string: Optional[str] = None):
        self._filepath = filepath

        self._motors: List[Motor] = []
        self._config = self._parse_config(config_string) if config_string else {}

        self._parse_registers()

    def _parse_config(self, config_string: str) -> Dict[str, Any]:
        config = {}
        entries = config_string.split('|')
        for entry in entries:
            if entry:
                key, value = entry.split(':')
                config[key.strip()] = self._convert_value(value.strip())
        return config

    def _convert_value(self, value: str) -> Union[int, float, bool, str]:
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            return value

    def _parse_registers(self) -> None:
        with open(self._filepath, 'r') as file:
            log_content = file.read()

        sections = re.split(self._section_pattern, log_content)

        # Detect only the latest dumps from the log (to ignore potential previous and outdated dumps)
        last_sections: Dict[str, int] = {}
        for i in range(1, len(sections), 2):
            stepper_name = 'stepper_' + sections[i].strip()
            last_sections[stepper_name] = i

        for stepper_name, index in last_sections.items():
            content = sections[index + 1]
            motor = Motor(stepper_name)

            # Apply general properties from config string
            for key, value in self._config.items():
                if stepper_name in key:
                    prop_key = key.replace(stepper_name + '_', '')
                    motor.set_property(prop_key, value)
                elif 'autotune' in key:
                    motor.set_property(key, value)

            # Parse TMC registers
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

            self._motors.append(motor)

    # Find and return the motor by its name
    def get_motor(self, motor_name: str) -> Optional[Motor]:
        for motor in self._motors:
            if motor._name == motor_name:
                return motor
        return None

    # Get all the motor list at once
    def get_motors(self) -> List[Motor]:
        return self._motors


# # Usage example:
#     config_string = "stepper_x_tmc:tmc2240|stepper_x_run_current:0.9|stepper_x_hold_current:0.9|stepper_y_tmc:tmc2240|stepper_y_run_current:0.9|stepper_y_hold_current:0.9|autotune_enabled:True|stepper_x_motor:ldo-35sth48-1684ah|stepper_x_voltage:|stepper_y_motor:ldo-35sth48-1684ah|stepper_y_voltage:|"
#     parser = MotorLogParser('/path/to/your/logfile.log', config_string)

#     stepper_x = parser.get_motor('stepper_x')
#     stepper_y = parser.get_motor('stepper_y')

#     print(stepper_x)
#     print(stepper_y)
