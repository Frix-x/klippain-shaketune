# Klippain Shake&Tune Module

This Klippain "Shake&Tune" repository is a standalone module from the [Klippain](https://github.com/Frix-x/klippain) ecosystem, designed to automate and calibrate the input shaper system on your Klipper 3D printer with a streamlined workflow and insightful vizualisations.

![logo banner](./docs/banner.png)

It operates in two steps:
  1. Utilizing specially tailored Klipper macros, it initiates tests on either the belts or the printer X/Y axis to measure the machine axes behavior. This is basically an automated call to the Klipper `TEST_RESONANCES` macro with custom parameters.
  2. Then a custom Python script is called to: 
     1. Generate insightful and improved graphs, aiding in parameter tuning for the Klipper `[input_shaper]` system (including best shaper choice, resonant frequency and damping ratio) or diagnosing and rectifying mechanical issues (like belt tension, defective bearings, etc..)
     2. Relocates the graphs and associated CSV files to your Klipper config folder for easy access via Mainsail/Fluidd to eliminate the need for SSH.
     3. Manages the folder by retaining only the most recent results (default setting of keeping the latest three sets).

Check out the **[detailed documentation of the Shake&Tune module here](./docs/README.md)**. You can also look at the documentation for each type of graph by directly clicking on them below to better understand your results and tune your machine!

| [Belts graph](./docs/macros/belts_tuning.md) | [Axis input shaper graphs](./docs/macros/axis_tuning.md) | [Vibrations graph](./docs/macros/vibrations_tuning.md) |
|:----------------:|:------------:|:---------------------:|
| [<img src="./docs/images/belts_example.png">](./docs/macros/belts_tuning.md) | [<img src="./docs/images/axis_example.png">](./docs/macros/axis_tuning.md) | [<img src="./docs/images/vibrations_example.png">](./docs/macros/vibrations_tuning.md) |

## Installation

Follow these steps to install the Shake&Tune module in your printer:
  1. Be sure to have a working accelerometer on your machine. You can follow the official [Measuring Resonances Klipper documentation](https://www.klipper3d.org/Measuring_Resonances.html) to configure one. Validate with an `ACCELEROMETER_QUERY` command that everything works correctly.
  1. Install the Shake&Tune package by running over SSH on your printer:
     ```bash
     wget -O - https://raw.githubusercontent.com/Frix-x/klippain-shaketune/main/install.sh | bash
     ```
  1. Then, append the following to your `printer.cfg` file and restart Klipper (if prefered, you can include only the needed macros: using `*.cfg` is a convenient way to include them all at once):
     ```
     [include K-ShakeTune/*.cfg]
     ```
  1. Finally, if you want to get automatic updates, add the following to your `moonraker.cfg` file:
     ```
     [update_manager Klippain-ShakeTune]
     type: git_repo
     path: ~/klippain_shaketune
     channel: beta
     origin: https://github.com/Frix-x/klippain-shaketune.git
     primary_branch: main
     managed_services: klipper
     install_script: install.sh
     ```


## Usage

Ensure your machine is homed, then invoke one of the following macros as needed:
  - `AXES_MAP_CALIBRATION` to automatically find Klipper's `axes_map` parameter for your accelerometer orientation (be careful, this is experimental for now).
  - `BELTS_SHAPER_CALIBRATION` for belt resonance graphs, useful for verifying belt tension and differential belt paths behavior.
  - `AXES_SHAPER_CALIBRATION` for input shaper graphs to mitigate ringing/ghosting by tuning Klipper's input shaper system.
  - `VIBRATIONS_CALIBRATION` for machine and motors vibration graphs, used to optimize your slicer speed profiles and TMC drivers parameters.
  - `EXCITATE_AXIS_AT_FREQ` to sustain a specific excitation frequency, useful to let you inspect and find out what is resonating.

For further insights on the usage of these macros and the generated graphs, refer to the [K-Shake&Tune module documentation](./docs/README.md).
