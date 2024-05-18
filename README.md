# Klipper Shake&Tune Module

This "Shake&Tune" repository is a standalone module from the [Klippain](https://github.com/Frix-x/klippain) ecosystem, designed to automate and calibrate the input shaper system on your Klipper 3D printer with a streamlined workflow and insightful vizualisations. This can be installed on any Klipper machine. It is not limited to those using Klippain.

![logo banner](./docs/banner.png)

It operates in two steps:
  1. Utilizing specially tailored Klipper macros, it initiates tests on either the belts or the printer X/Y axis to measure the machine axes behavior. This is basically an automated call to the Klipper `TEST_RESONANCES` macro with custom parameters.
  2. Then a custom Python script is called to: 
     1. Generate insightful and improved graphs, aiding in parameter tuning for the Klipper `[input_shaper]` system (including best shaper choice, resonant frequency and damping ratio) or diagnosing and rectifying mechanical issues (like belt tension, defective bearings, etc..)
     2. Relocates the graphs and associated CSV files to your Klipper config folder for easy access via Mainsail/Fluidd to eliminate the need for SSH.
     3. Manages the folder by retaining only the most recent results (default setting of keeping the latest three sets).

Check out the **[detailed documentation of the Shake&Tune module here](./docs/README.md)**. You can also look at the documentation for each type of graph by directly clicking on them below to better understand your results and tune your machine!

| [Belts graph](./docs/macros/belts_tuning.md) | [Axis input shaper graphs](./docs/macros/axis_tuning.md) | [Vibrations graph](./docs/macros/vibrations_profile.md) |
|:----------------:|:------------:|:---------------------:|
| [<img src="./docs/images/belts_example.png">](./docs/macros/belts_tuning.md) | [<img src="./docs/images/axis_example.png">](./docs/macros/axis_tuning.md) | [<img src="./docs/images/vibrations_example.png">](./docs/macros/vibrations_profile.md) |

  > **Note**:
  >
  > Be aware that Shake&Tune uses the [Gcode shell command plugin](https://github.com/dw-0/kiauh/blob/master/docs/gcode_shell_command.md) under the hood to call the Python scripts that generate the graphs. While my scripts should be safe, the Gcode shell command plugin also has great potential for abuse if not used carefully for other purposes, since it opens shell access from Klipper.

## Installation

Follow these steps to install the Shake&Tune module in your printer:
  1. Be sure to have a working accelerometer on your machine and a `[resonance_tester]` section defined. You can follow the official [Measuring Resonances Klipper documentation](https://www.klipper3d.org/Measuring_Resonances.html) to configure it.
  1. Install the Shake&Tune package by running over SSH on your printer:
     ```bash
     wget -O - https://raw.githubusercontent.com/Frix-x/klippain-shaketune/main/install.sh | bash
     ```
  1. Then, append the following to your `printer.cfg` file and restart Klipper (if prefered, you can include only the needed macros: using `*.cfg` is a convenient way to include them all at once):
     ```
     [include K-ShakeTune/*.cfg]
     ```

## Usage

Ensure your machine is homed, then invoke one of the following macros as needed:
  - `AXES_MAP_CALIBRATION` to automatically find Klipper's `axes_map` parameter for your accelerometer orientation (be careful, this is experimental for now and known to give bad results).
  - `COMPARE_BELTS_RESPONSES` for a differential belt resonance graph, useful for checking relative belt tensions and belt path behaviors on a CoreXY printer.
  - `AXES_SHAPER_CALIBRATION` for standard input shaper graphs, used to mitigate ringing/ghosting by tuning Klipper's input shaper filters.
  - `CREATE_VIBRATIONS_PROFILE` for vibrations graphs as a function of toolhead direction and speed, used to find problematic ranges where the printer could be exposed to more VFAs and optimize your slicer speed profiles and TMC driver parameters.
  - `EXCITATE_AXIS_AT_FREQ` to maintain a specific excitation frequency, useful to inspect and find out what is resonating.

For further insights on the usage of these macros and the generated graphs, refer to the [K-Shake&Tune module documentation](./docs/README.md).
