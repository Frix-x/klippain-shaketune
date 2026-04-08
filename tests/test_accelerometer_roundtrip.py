import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import shaketune.helpers.accelerometer as accelerometer_module
from shaketune.helpers.accelerometer import MeasurementsManager


class _FakeReactor:
    def monotonic(self):
        return time.monotonic()

    def pause(self, waketime):
        now = time.monotonic()
        if waketime > now:
            time.sleep(waketime - now)
        return time.monotonic()


class _ThreadProcess:
    def __init__(self, target, args=(), daemon=False):
        self._thread = threading.Thread(target=target, args=args, daemon=daemon)

    def start(self):
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()


class MeasurementsManagerRoundTripTest(unittest.TestCase):
    def test_loaded_stdata_measurements_can_be_saved_again(self):
        initial_samples = [
            (0.0, 1.0, 2.0, 3.0),
            (0.01, 1.5, 2.5, 3.5),
            (0.02, 2.0, 3.0, 4.0),
        ]
        additional_samples = [
            (0.0, 4.0, 5.0, 6.0),
            (0.01, 4.5, 5.5, 6.5),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            first_file = tmpdir_path / 'first.stdata'
            second_file = tmpdir_path / 'second.stdata'

            with patch.object(accelerometer_module, 'Process', _ThreadProcess):
                writer = MeasurementsManager(chunk_size=1, k_reactor=_FakeReactor(), stdata_filename=first_file)
                writer.add_measurement('first', initial_samples)
                writer.save_stdata()

                roundtrip = MeasurementsManager(chunk_size=1, k_reactor=_FakeReactor(), stdata_filename=second_file)
                loaded_measurements = roundtrip.load_from_stdata(first_file)

                self.assertEqual(len(loaded_measurements), 1)
                self.assertIsInstance(loaded_measurements[0]['samples'], np.ndarray)
                np.testing.assert_allclose(loaded_measurements[0]['samples'], np.asarray(initial_samples))

                # Force the loaded NumPy-backed measurement back through the writer queue.
                roundtrip.add_measurement('second', additional_samples)
                roundtrip.save_stdata()

                reloaded = MeasurementsManager(chunk_size=1).load_from_stdata(second_file)

                self.assertEqual([measurement['name'] for measurement in reloaded], ['first', 'second'])
                self.assertIsInstance(reloaded[0]['samples'], np.ndarray)
                self.assertIsInstance(reloaded[1]['samples'], np.ndarray)
                np.testing.assert_allclose(reloaded[0]['samples'], np.asarray(initial_samples))
                np.testing.assert_allclose(reloaded[1]['samples'], np.asarray(additional_samples))


if __name__ == '__main__':
    unittest.main()
