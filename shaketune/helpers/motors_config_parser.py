# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: motors_config_parser.py
# Description: Contains classes to retrieve motor information and extract relevant data
#              from the Klipper configuration and TMC registers.


from typing import Any, Dict, List, Optional

TRINAMIC_DRIVERS = ['tmc2130', 'tmc2208', 'tmc2209', 'tmc2240', 'tmc2660', 'tmc5160']
MOTORS = ['stepper_x', 'stepper_y', 'stepper_x1', 'stepper_y1', 'stepper_z', 'stepper_z1', 'stepper_z2', 'stepper_z3']
RELEVANT_TMC_REGISTERS = ['CHOPCONF', 'PWMCONF', 'COOLCONF', 'TPWMTHRS', 'TCOOLTHRS']


class Motor:
    def __init__(self, name: str):
        self.name: str = name
        self._registers: Dict[str, Dict[str, Any]] = {}
        self._config: Dict[str, Any] = {}

    def set_register(self, register: str, value_dict: dict) -> None:
        # First we filter out entries with a value of 0 to avoid having too much uneeded data
        value_dict = {k: v for k, v in value_dict.items() if v != 0}

        # Special parsing for CHOPCONF to extract meaningful values
        if register == 'CHOPCONF':
            # Add intpol=0 if missing from the register dump to force printing it as it's important
            if 'intpol' not in value_dict:
                value_dict['intpol'] = '0'
            # Remove the microsteps entry as the format here is not easy to read and
            # it's already read in the correct format directly from the Klipper config
            if 'mres' in value_dict:
                del value_dict['mres']

        # Special parsing for CHOPCONF to avoid pwm_ before each values
        if register == 'PWMCONF':
            new_value_dict = {}
            for key, val in value_dict.items():
                if key.startswith('pwm_'):
                    key = key[4:]
                new_value_dict[key] = val
            value_dict = new_value_dict

        # Then gets merged all the thresholds into the same THRS virtual register
        if register in {'TPWMTHRS', 'TCOOLTHRS'}:
            existing_thrs = self._registers.get('THRS', {})
            merged_values = {**existing_thrs, **value_dict}
            self._registers['THRS'] = merged_values
        else:
            self._registers[register] = value_dict

    def get_register(self, register: str) -> Optional[Dict[str, Any]]:
        return self._registers.get(register)

    def get_registers(self) -> Dict[str, Dict[str, Any]]:
        return self._registers

    def set_config(self, field: str, value: Any) -> None:
        self._config[field] = value

    def get_config(self, field: str) -> Optional[Any]:
        return self._config.get(field)

    def __str__(self):
        return f'Stepper: {self.name}\nKlipper config: {self._config}\nTMC Registers: {self._registers}'

    # Return the other motor config and registers that are different from the current motor
    def compare_to(self, other: 'Motor') -> Optional[Dict[str, Dict[str, Any]]]:
        differences = {'config': {}, 'registers': {}}

        # Compare Klipper config
        all_keys = self._config.keys() | other._config.keys()
        for key in all_keys:
            val1 = self._config.get(key)
            val2 = other._config.get(key)
            if val1 != val2:
                differences['config'][key] = val2

        # Compare TMC registers
        all_keys = self._registers.keys() | other._registers.keys()
        for key in all_keys:
            reg1 = self._registers.get(key, {})
            reg2 = other._registers.get(key, {})
            if reg1 != reg2:
                reg_diffs = {}
                sub_keys = reg1.keys() | reg2.keys()
                for sub_key in sub_keys:
                    reg_val1 = reg1.get(sub_key)
                    reg_val2 = reg2.get(sub_key)
                    if reg_val1 != reg_val2:
                        reg_diffs[sub_key] = reg_val2
                if reg_diffs:
                    differences['registers'][key] = reg_diffs

        # Clean up: remove empty sections if there are no differences
        if not differences['config']:
            del differences['config']
        if not differences['registers']:
            del differences['registers']

        return None if not differences else differences


class MotorsConfigParser:
    def __init__(self, config, motors: List[str] = MOTORS, drivers: List[str] = TRINAMIC_DRIVERS):
        self._printer = config.get_printer()

        self._motors: List[Motor] = []

        if motors is not None:
            for motor_name in motors:
                for driver in drivers:
                    tmc_object = self._printer.lookup_object(f'{driver} {motor_name}', None)
                    if tmc_object is None:
                        continue
                    motor = self._create_motor(motor_name, driver, tmc_object)
                    self._motors.append(motor)

        pconfig = self._printer.lookup_object('configfile')
        self.kinematics = pconfig.status_raw_config['printer']['kinematics']

    # Create a Motor object with the given name, driver and TMC object
    # and fill it with the relevant configuration and registers
    def _create_motor(self, motor_name: str, driver: str, tmc_object: Any) -> Motor:
        motor = Motor(motor_name)
        motor.set_config('tmc', driver)
        self._parse_klipper_config(motor, tmc_object)
        self._parse_tmc_registers(motor, tmc_object)
        return motor

    def _parse_klipper_config(self, motor: Motor, tmc_object: Any) -> None:
        # The TMCCommandHelper isn't a direct member of the TMC object... but we can still get it this way
        tmc_cmdhelper = tmc_object.get_status.__self__

        motor_currents = tmc_cmdhelper.current_helper.get_current()
        motor.set_config('run_current', motor_currents[0])
        motor.set_config('hold_current', motor_currents[1])

        pconfig = self._printer.lookup_object('configfile')
        motor.set_config('microsteps', int(pconfig.status_raw_config[motor.name]['microsteps']))

        autotune_object = self._printer.lookup_object(f'autotune_tmc {motor.name}', None)
        if autotune_object is not None:
            motor.set_config('autotune_enabled', True)
            motor.set_config('motor', autotune_object.motor)
            motor.set_config('voltage', autotune_object.voltage)
            motor.set_config('pwm_freq_target', autotune_object.pwm_freq_target)
        else:
            motor.set_config('autotune_enabled', False)

    def _parse_tmc_registers(self, motor: Motor, tmc_object: Any) -> None:
        # The TMCCommandHelper isn't a direct member of the TMC object... but we can still get it this way
        tmc_cmdhelper = tmc_object.get_status.__self__

        for register in RELEVANT_TMC_REGISTERS:
            val = tmc_cmdhelper.fields.registers.get(register)
            if (val is not None) and (register not in tmc_cmdhelper.read_registers):
                # write-only register
                fields_string = self._extract_register_values(tmc_cmdhelper, register, val)
            elif register in tmc_cmdhelper.read_registers:
                # readable register
                val = tmc_cmdhelper.mcu_tmc.get_register(register)
                if tmc_cmdhelper.read_translate is not None:
                    register, val = tmc_cmdhelper.read_translate(register, val)
                fields_string = self._extract_register_values(tmc_cmdhelper, register, val)

            motor.set_register(register, fields_string)

    def _extract_register_values(self, tmc_cmdhelper, register, val):
        # Provide a dictionary of register values
        reg_fields = tmc_cmdhelper.fields.all_fields.get(register, {})
        reg_fields = sorted([(mask, name) for name, mask in reg_fields.items()])
        fields = {}
        for _, field_name in reg_fields:
            field_value = tmc_cmdhelper.fields.get_field(field_name, val, register)
            fields[field_name] = field_value
        return fields

    # Find and return the motor by its name
    def get_motor(self, motor_name: str) -> Optional[Motor]:
        return next((motor for motor in self._motors if motor.name == motor_name), None)

    # Get all the motor list at once
    def get_motors(self) -> List[Motor]:
        return self._motors
