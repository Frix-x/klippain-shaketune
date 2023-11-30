# Klippain Shake&Tune module documentation

![](./banner_long.png)

### Detailed documentation

Before running any tests, you should first run the `AXES_MAP_CALIBRATION` macro to detect the and set Klipper's accelerometer axes_map parameter and validate that your accelerometer is working correctly and properly mounted. Then, check out **[input shaping and tuning generalities](./is_tuning_generalities.md)** to understand how it all works and what to look for when taking these measurements.

Finally look at the documentation for each type of graph by clicking on them below tu run the tests and better understand your results to tune your machine!

| [Belts graph](./macros/belts_tuning.md) | [Axis input shaper graphs](./macros/axis_tuning.md) | [Vibrations graph](./macros/vibrations_tuning.md) |
|:----------------:|:------------:|:---------------------:|
| [<img src="./images/belts_example.png">](./macros/belts_tuning.md) | [<img src="./images/axis_example.png">](./macros/axis_tuning.md) | [<img src="./images/vibrations_example.png">](./macros/vibrations_tuning.md) |


### Complementary ressources

  - [Sineos post](https://klipper.discourse.group/t/interpreting-the-input-shaper-graphs/9879) in the Klipper knowledge base
