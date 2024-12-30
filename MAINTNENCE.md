## When a new stable version of klipper is released:

In `.github/workflows/test.yaml`, update `jobs.klippy_testing.strategy.matrix.klipper_version` to include the new version.

## When a new version of python becomes supported by klipper

In `.github/workflows/test.yaml`, update `jobs.klippy_testing.strategy.matrix.python_version` to include the new version.
