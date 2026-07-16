#!/usr/bin/env python3
"""Regression test for the shaper.vals <-> calibration_data.freqs length mismatch.

Background: ShaperComputation.compute() (shaketune/graph_creators/computations/
shaper_computation.py) used to assume that on older Klipper/Kalico builds (ones
whose CalibrationResult has no `freq_bins` field), `shaper.vals` is already
truncated to exactly match `calibration_data.freqs`. That assumption breaks
whenever Kalico's own `fit_shaper` inflates its internal max_freq past what
was passed in (`max_freq = max(max_freq, test_freqs.max())`), which happens
whenever the requested max_freq is lower than the shaper search's own
frequency ceiling (MAX_SHAPER_FREQ, 150Hz on KalicoCrew/kalico). The result
was an unaligned array reaching matplotlib as an opaque "x and y must have
same first dimension" crash during graph generation.

This script drives the real ShaperComputation against a real Kalico/Klipper
checkout with a synthetic capture, and asserts every shaper's `vals` is
exactly aligned with `calibration_data.freqs` at a range of max_freq values
-- including ones below and above MAX_SHAPER_FREQ, since the bug is only
observable when the requested max_freq undercuts it.

Usage:
    python test_shaper_length_mismatch.py --klipper-dir /path/to/kalico/checkout
"""

import argparse
import math
import os
import random
import sys
from importlib import import_module
from pathlib import Path


def build_synthetic_samples(sample_rate=3200.0, duration=2.0, seed=1234):
    """A resonance-like raw accelerometer capture: dominant peak ~190Hz plus
    broadband content out past 300Hz, so there's real data to truncate."""
    random.seed(seed)
    n = int(sample_rate * duration)
    samples = []
    for i in range(n):
        t = i / sample_rate
        decay = 1.0 if i < n * 0.9 else (n - i) / (n * 0.1)
        x = (
            3000.0 * decay * math.sin(2 * math.pi * 190.0 * t)
            + 400.0 * math.sin(2 * math.pi * 45.0 * t)
            + 150.0 * math.sin(2 * math.pi * 260.0 * t)
            + random.uniform(-50, 50)
        )
        y = 200.0 * math.sin(2 * math.pi * 97.0 * t) + random.uniform(-30, 30)
        z = random.uniform(-20, 20)
        samples.append((t, x, y, z))
    return samples


def load_kalico_shaper_calibrate(klipper_dir):
    """Mirrors shaketune.cli.load_klipper_module(): point sys.modules at the
    real installed shaper_calibrate/shaper_defs so ShaperComputation runs
    against actual firmware code, not a mock."""
    os.environ['SHAKETUNE_IN_CLI'] = '1'
    kdir = os.path.expanduser(klipper_dir)
    sys.path.append(os.path.join(kdir, 'klippy'))
    sys.modules['shaper_calibrate'] = import_module('.shaper_calibrate', 'extras')
    sys.modules['shaper_defs'] = import_module('.shaper_defs', 'extras')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--klipper-dir',
        required=True,
        help='Path to a Kalico/Klipper checkout (needs klippy/extras/shaper_calibrate.py)',
    )
    parser.add_argument(
        '--max-freqs',
        type=float,
        nargs='+',
        default=[80.0, 100.0, 149.0, 150.0, 151.0, 200.0, 250.0, 300.0],
        help='max_freq values to test; below/around/above MAX_SHAPER_FREQ (150 on KalicoCrew/kalico) is the interesting range',
    )
    args = parser.parse_args()

    load_kalico_shaper_calibrate(args.klipper_dir)

    # Import after sys.path/sys.modules are set up, and after this script's own
    # directory (the shaketune repo root) is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from shaketune.graph_creators.computations.shaper_computation import ShaperComputation

    samples = build_synthetic_samples()
    measurement = {'name': 'synthetic_x', 'samples': samples}

    failures = []
    for max_freq in args.max_freqs:
        computation = ShaperComputation(
            measurements=[measurement],
            max_smoothing=None,
            scv=5.0,
            max_freq=max_freq,
            test_params=None,
            max_scale=None,
            st_version='test',
        )
        result = computation.compute()
        n_expected = len(result.calibration_data.freqs)
        for shaper in result.shaper_table_data['shapers']:
            n_actual = len(shaper['vals'])
            status = 'OK' if n_actual == n_expected else 'MISMATCH'
            print(
                f'max_freq={max_freq:>6.1f}  {shaper["type"]:<12} vals={n_actual:<5} expected={n_expected:<5} {status}'
            )
            if n_actual != n_expected:
                failures.append((max_freq, shaper['type'], n_actual, n_expected))

    print()
    if failures:
        print(f'FAILED: {len(failures)} shaper/max_freq combination(s) had mismatched array lengths:')
        for max_freq, name, actual, expected in failures:
            print(f'  max_freq={max_freq}: {name} vals={actual} != freqs={expected}')
        sys.exit(1)
    else:
        print(
            f'PASSED: all shapers matched calibration_data.freqs length across {len(args.max_freqs)} max_freq value(s).'
        )


if __name__ == '__main__':
    main()
