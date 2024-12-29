# Shake&Tune: 3D printer analysis tools
#
# Adapted from Klipper's original resonance_tester.py file by Dmitry Butyugin <dmbutyugin@google.com>
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: resonance_test.py
# Description: Contains functions to test the resonance frequency of the printer and its components
#              by vibrating the toolhead in specific axis directions. This derive a bit from Klipper's
#              implementation as there are a couple of changes:
#                1. Original code doesn't use euclidean distance with projection for the coordinates calculation.
#                   The new approach implemented here ensures that the vector's total length remains constant (= L),
#                   regardless of the direction components. It's especially important when the direction vector
#                   involves combinations of movements along multiple axes like for the diagonal belt tests.
#                2. Original code doesn't allow Z axis movements that was added in order to test the Z axis resonance
#                   or CoreXZ belts frequency profiles as well.
#                3. There is the possibility to do a real static frequency test by specifying a duration and a
#                   fixed frequency to maintain.


import math
from collections import namedtuple

from ..helpers.console_output import ConsoleOutput

testParams = namedtuple(
    'testParams', ['mode', 'min_freq', 'max_freq', 'accel_per_hz', 'hz_per_sec', 'sweeping_accel', 'sweeping_period']
)


# This class is used to generate the base vibration test sequences
# Note: it's almost untouched from the original Klipper code from Dmitry Butyugin
class BaseVibrationGenerator:
    def __init__(self, min_freq, max_freq, accel_per_hz, hz_per_sec):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.accel_per_hz = accel_per_hz
        self.hz_per_sec = hz_per_sec
        self.freq_start = min_freq
        self.freq_end = max_freq

    def prepare_test(self, freq_start=None, freq_end=None, accel_per_hz=None, hz_per_sec=None):
        if freq_start is not None:
            self.freq_start = freq_start
        if freq_end is not None:
            self.freq_end = freq_end
        if accel_per_hz is not None:
            self.accel_per_hz = accel_per_hz
        if hz_per_sec is not None:
            self.hz_per_sec = hz_per_sec

    def get_max_freq(self):
        return self.freq_end

    def gen_test(self):
        freq = self.freq_start
        result = []
        sign = 1.0
        time = 0.0
        while freq <= self.freq_end + 0.000001:
            t_seg = 0.25 / freq
            accel = self.accel_per_hz * freq
            time += t_seg
            result.append((time, sign * accel, freq))
            time += t_seg
            result.append((time, -sign * accel, freq))
            freq += 2.0 * t_seg * self.hz_per_sec
            sign = -sign
        return result


# This class is a derivative of BaseVibrationGenerator that adds slow sweeping acceleration to the test sequences (new style)
# Note: it's almost untouched from the original Klipper code from Dmitry Butyugin
class SweepingVibrationGenerator(BaseVibrationGenerator):
    def __init__(self, min_freq, max_freq, accel_per_hz, hz_per_sec, sweeping_accel=400.0, sweeping_period=1.2):
        super().__init__(min_freq, max_freq, accel_per_hz, hz_per_sec)
        self.sweeping_accel = sweeping_accel
        self.sweeping_period = sweeping_period

    def prepare_test(
        self,
        freq_start=None,
        freq_end=None,
        accel_per_hz=None,
        hz_per_sec=None,
        sweeping_accel=None,
        sweeping_period=None,
    ):
        super().prepare_test(freq_start, freq_end, accel_per_hz, hz_per_sec)
        if sweeping_accel is not None:
            self.sweeping_accel = sweeping_accel
        if sweeping_period is not None:
            self.sweeping_period = sweeping_period

    def gen_test(self):
        base_seq = super().gen_test()
        if not self.sweeping_period:
            # If sweeping_period == 0, just return base sequence (old style pulse-only test)
            return base_seq

        accel_fraction = math.sqrt(2.0) * 0.125
        t_rem = self.sweeping_period * accel_fraction
        sweeping_accel = self.sweeping_accel
        result = []
        last_t = 0.0
        sig = 1.0
        accel_fraction += 0.25

        for next_t, accel, freq in base_seq:
            t_seg = next_t - last_t
            while t_rem <= t_seg:
                last_t += t_rem
                result.append((last_t, accel + sweeping_accel * sig, freq))
                t_seg -= t_rem
                t_rem = self.sweeping_period * accel_fraction
                accel_fraction = 0.5
                sig = -sig
            t_rem -= t_seg
            result.append((next_t, accel + sweeping_accel * sig, freq))
            last_t = next_t

        return result


# This class is a specialized generator for maintaining a single fixed frequency of vibration for a given duration.
# For simplicity, it uses the same old-style pulse pattern as the base class.
class StaticFrequencyVibrationGenerator(BaseVibrationGenerator):
    def __init__(self, freq, accel_per_hz, duration):
        # For a static frequency, min_freq = max_freq = freq, hz_per_sec doesn't matter.
        super().__init__(freq, freq, accel_per_hz, hz_per_sec=None)
        self.duration = duration

    def gen_test(self):
        freq = self.freq_start  # same as self.freq_end since static
        t_seg = 0.25 / freq
        accel = self.accel_per_hz * freq
        sign = 1.0
        time = 0.0
        result = []
        # We'll produce pulses until we exceed the specified duration
        while time < self.duration:
            time += t_seg
            if time > self.duration:
                break
            result.append((time, sign * accel, freq))

            time += t_seg
            if time > self.duration:
                break
            result.append((time, -sign * accel, freq))
            sign = -sign

        return result


# This class manages and executes resonance tests, handling both old and new Klipper logic
class ResonanceTestManager:
    def __init__(self, toolhead, gcode, res_tester):
        self.toolhead = toolhead
        self.gcode = gcode
        self.res_tester = res_tester
        self.reactor = self.toolhead.reactor

    @property
    def is_old_klipper(self):
        return hasattr(self.res_tester, 'test')

    def get_parameters(self):
        if self.is_old_klipper:
            return (
                self.res_tester.test.min_freq,
                self.res_tester.test.max_freq,
                self.res_tester.test.accel_per_hz,
                self.res_tester.test.hz_per_sec,
                0.0,  # sweeping_period=0 to force the old style pulse-only test
                None,  # sweeping_accel unused in old style pulse-only test
            )
        else:
            return (
                self.res_tester.generator.vibration_generator.min_freq,
                self.res_tester.generator.vibration_generator.max_freq,
                self.res_tester.generator.vibration_generator.accel_per_hz,
                self.res_tester.generator.vibration_generator.hz_per_sec,
                self.res_tester.generator.sweeping_period,
                self.res_tester.generator.sweeping_accel,
            )

    def vibrate_axis(
        self, axis_direction, min_freq=None, max_freq=None, hz_per_sec=None, accel_per_hz=None
    ) -> testParams:
        base_min_freq, base_max_freq, base_aph, base_hps, base_s_period, base_s_accel = self.get_parameters()

        final_min_f = min_freq if min_freq is not None else base_min_freq
        final_max_f = max_freq if max_freq is not None else base_max_freq
        final_aph = accel_per_hz if accel_per_hz is not None else base_aph
        final_hps = hz_per_sec if hz_per_sec is not None else base_hps
        s_period = base_s_period
        s_accel = base_s_accel

        if s_period == 0 or self.is_old_klipper:
            ConsoleOutput.print('Using pulse-only test')
            gen = BaseVibrationGenerator(final_min_f, final_max_f, final_aph, final_hps)
            test_params = testParams('PULSE-ONLY', final_min_f, final_max_f, final_aph, final_hps, None, None)
        else:
            ConsoleOutput.print('Using pulse test with additional sweeping')
            gen = SweepingVibrationGenerator(final_min_f, final_max_f, final_aph, final_hps, s_accel, s_period)
            test_params = testParams('SWEEPING', final_min_f, final_max_f, final_aph, final_hps, s_accel, s_period)

        test_seq = gen.gen_test()
        self._run_test_sequence(axis_direction, test_seq)
        self.toolhead.wait_moves()
        return test_params

    def vibrate_axis_at_static_freq(self, axis_direction, freq, duration, accel_per_hz) -> testParams:
        gen = StaticFrequencyVibrationGenerator(freq, accel_per_hz, duration)
        test_seq = gen.gen_test()
        self._run_test_sequence(axis_direction, test_seq)
        self.toolhead.wait_moves()
        return testParams('static', freq, freq, accel_per_hz, None, None, None)

    def _run_test_sequence(self, axis_direction, test_seq):
        toolhead = self.toolhead
        gcode = self.gcode
        reactor = self.reactor
        systime = reactor.monotonic()
        toolhead_info = toolhead.get_status(systime)
        X, Y, Z, E = toolhead.get_position()

        old_max_accel = toolhead_info['max_accel']

        # Set velocity limits
        max_accel = max(abs(a) for _, a, _ in test_seq) if test_seq else old_max_accel
        if 'minimum_cruise_ratio' in toolhead_info:  # minimum_cruise_ratio found: Klipper >= v0.12.0-239
            old_mcr = toolhead_info['minimum_cruise_ratio']
            gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={max_accel} MINIMUM_CRUISE_RATIO=0')
        else:  # minimum_cruise_ratio not found: Klipper < v0.12.0-239
            old_mcr = None
            gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={max_accel}')

        # Disable input shaper if present
        input_shaper = self.toolhead.printer.lookup_object('input_shaper', None)
        if input_shaper is not None:
            input_shaper.disable_shaping()
            ConsoleOutput.print('Disabled [input_shaper] for resonance testing')

        normalized_direction = self._normalize_direction(axis_direction)
        last_v = 0.0
        last_t = 0.0
        last_v2 = 0.0
        last_freq = 0.0

        for next_t, accel, freq in test_seq:
            t_seg = next_t - last_t
            toolhead.cmd_M204(gcode.create_gcode_command('M204', 'M204', {'S': abs(accel)}))
            v = last_v + accel * t_seg
            abs_v = abs(v)
            if abs_v < 1e-6:
                v = abs_v = 0.0
            abs_last_v = abs(last_v)

            v2 = v * v
            half_inv_accel = 0.5 / accel if accel != 0 else 0.0
            d = (v2 - last_v2) * half_inv_accel if accel != 0 else 0.0
            dX, dY, dZ = self._project_distance(d, normalized_direction)
            nX, nY, nZ = X + dX, Y + dY, Z + dZ

            if not self.is_old_klipper:
                toolhead.limit_next_junction_speed(abs_last_v)

            # If direction changed sign, must pass through zero velocity
            if v * last_v < 0:
                d_decel = -last_v2 * half_inv_accel if accel != 0 else 0.0
                decel_x, decel_y, decel_z = self._project_distance(d_decel, normalized_direction)
                toolhead.move([X + decel_x, Y + decel_y, Z + decel_z, E], abs_last_v)
                toolhead.move([nX, nY, nZ, E], abs_v)
            else:
                toolhead.move([nX, nY, nZ, E], max(abs_v, abs_last_v))

            if math.floor(freq) > math.floor(last_freq):
                ConsoleOutput.print(f'Testing frequency: {freq:.0f} Hz')
                reactor.pause(reactor.monotonic() + 0.01)

            X, Y, Z = nX, nY, nZ
            last_t = next_t
            last_v = v
            last_v2 = v2
            last_freq = freq

        # Decelerate if needed
        if last_v != 0.0:
            d_decel = -0.5 * last_v2 / old_max_accel if old_max_accel != 0 else 0
            ddX, ddY, ddZ = self._project_distance(d_decel, normalized_direction)
            toolhead.cmd_M204(gcode.create_gcode_command('M204', 'M204', {'S': old_max_accel}))
            toolhead.move([X + ddX, Y + ddY, Z + ddZ, E], abs(last_v))

        # Restore the previous acceleration values
        if old_mcr is not None:  # minimum_cruise_ratio found: Klipper >= v0.12.0-239
            gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_max_accel} MINIMUM_CRUISE_RATIO={old_mcr}')
        else:  # minimum_cruise_ratio not found: Klipper < v0.12.0-239
            gcode.run_script_from_command(f'SET_VELOCITY_LIMIT ACCEL={old_max_accel}')

        # Re-enable input shaper if disabled
        if input_shaper is not None:
            input_shaper.enable_shaping()
            ConsoleOutput.print('Re-enabled [input_shaper]')

    @staticmethod
    def _normalize_direction(direction):
        magnitude = math.sqrt(sum(c * c for c in direction))
        if magnitude < 1e-12:
            raise ValueError('Invalid axis direction: zero magnitude')
        return tuple(c / magnitude for c in direction)

    @staticmethod
    def _project_distance(distance, normalized_direction):
        return (
            normalized_direction[0] * distance,
            normalized_direction[1] * distance,
            normalized_direction[2] * distance,
        )


def vibrate_axis(
    toolhead, gcode, axis_direction, min_freq, max_freq, hz_per_sec, accel_per_hz, res_tester
) -> testParams:
    manager = ResonanceTestManager(toolhead, gcode, res_tester)
    return manager.vibrate_axis(axis_direction, min_freq, max_freq, hz_per_sec, accel_per_hz)


def vibrate_axis_at_static_freq(toolhead, gcode, axis_direction, freq, duration, accel_per_hz) -> testParams:
    manager = ResonanceTestManager(toolhead, gcode, None)
    return manager.vibrate_axis_at_static_freq(axis_direction, freq, duration, accel_per_hz)
