# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: compat.py
# Description: Handles compatibility with different versions of Klipper.

from collections import namedtuple

ResTesterConfig = namedtuple('ResTesterConfig', ['default_min_freq', 'default_max_freq', 'default_accel_per_hz', 'test_points'])

def res_tester_config(config) -> ResTesterConfig:
    printer = config.get_printer()
    res_tester = printer.lookup_object('resonance_tester')

    # Get the default values for the frequency range and the acceleration per Hz
    if hasattr(res_tester, 'test'):
        # Old Klipper code (before Dec 6, 2024: https://github.com/Klipper3d/klipper/commit/16b4b6b302ac3ffcd55006cd76265aad4e26ecc8)
        default_min_freq = res_tester.test.min_freq
        default_max_freq = res_tester.test.max_freq
        default_accel_per_hz = res_tester.test.accel_per_hz
        test_points = res_tester.test.get_start_test_points()
    else:
        # New Klipper code (after Dec 6, 2024) with the sweeping test
        default_min_freq = res_tester.generator.vibration_generator.min_freq
        default_max_freq = res_tester.generator.vibration_generator.max_freq
        default_accel_per_hz = res_tester.generator.vibration_generator.accel_per_hz
        test_points = res_tester.probe_points

    return ResTesterConfig(default_min_freq, default_max_freq, default_accel_per_hz, test_points)
