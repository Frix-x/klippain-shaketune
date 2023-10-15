# KISSY: Klippain Input Shaper SYstem

KISSY *Klippain Input Shaper System* is a module from the [Klippain](https://github.com/Frix-x/klippain) ecosystem, designed to automate and calibrate the input shaper system on your Klipper 3D printer with a streamlined workflow and insightful vizualisations.

![KISSY](./docs/kissy.png)

It operates in two steps:

  1. Utilizing specially tailored Klipper macros, it initiates tests on either the belts or the printer X/Y axis to measure the machine axes behavior. This is basically an automated call to the Klipper `TEST_RESONANCES` macro with custom parameters.
  
  2. Then a custom Python script is called to: 
     a. Generate insightful and improved graphs, aiding in parameter tuning for the Klipper `[input_shaper]` system (including best shaper choice, resonant frequency and damping ratio) or diagnosing and rectifying mechanical issues (like belt tension, defective bearings, etc..)
     b. Relocates the graphs and associated CSV files to your Klipper config folder for easy access via Mainsail/Fluidd to eliminate the need for SSH.
     c. Manages the folder by retaining only the most recent results (default setting of keeping the latest three sets).

> **Note**:
> 
> KISSY is part of the [Klippain](https://github.com/Frix-x/klippain) ecosystem. If you already have a full Klippain installation on your machine, no additional installation is required for you!

If needed, refer to [my IS graphs documentation](./docs/input_shaper.md) for tips on interpreting the generated graphs.

| Belts graphs | X graphs | Y graphs | Vibrations measurement |
|:----------------:|:------------:|:------------:|:---------------------:|
| ![](./docs/images/resonances_belts_example.png) | ![](./docs/images/resonances_x_example.png) | ![](./docs/images/resonances_y_example.png) | ![](./docs/images/vibrations_example.png) |

## Installation

For those not using the full [Klippain](https://github.com/Frix-x/klippain), follow these steps to integrate KISSY in your setup:
  1. Add the [KISSY folder](./KISSY/) and its contents to the root of your config directory (e.g., `~/printer_data/config/`).
  2. Ensure the `gcode_shell_command.py` Klipper extension is installed. Use the advanced section of [KIAUH](https://github.com/dw-0/kiauh) for a straightforward installation.
  3. Make the scripts executable via SSH within the folder (`cd ~/printer_data/config/KISSY/scripts`):
     ```bash
     chmod +x ./is_workflow.py
     chmod +x ./graph_belts.py
     chmod +x ./graph_shaper.py
     chmod +x ./graph_vibrations.py
     ```
  4. Append the following to your `printer.cfg` file:
     ```
     [include KISSY/*.cfg]
     ```

## Usage

Ensure your machine is homed, then invoke one of the following macros as needed:
  - `BELTS_SHAPER_CALIBRATION` for belt resonance graphs, useful for verifying belt tension and differential belt paths behavior.
  - `AXES_SHAPER_CALIBRATION` for input shaper graphs to mitigate ringing/ghosting by tuning Klipper's `[input_shaper]` system.
  - `VIBRATIONS_CALIBRATION` for machine vibration graphs to optimize your slicer speed profiles.
  - `EXCITATE_AXIS_AT_FREQ` to sustain a specific excitation frequency, useful to let you inspect and find out what is resonating.

Retrieve the results from the results folder, accessible directly via Mainsail/Fluidd WebUI. For further insight on reading the results, refer to my documentation on [interpreting the IS graphs](./docs/input_shaper.md).
