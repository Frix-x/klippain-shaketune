# Vibrations measurements

The `VIBRATIONS_CALIBRATION` macro helps you to identify the speed settings that exacerbate the vibrations of the machine (ie. where the frame and motors resonate badly). This will help you to find the clean speed ranges where the machine is more silent and less prone to vertical fine artifacts on the prints.

  > **Warning**
  >
  > You will first need to calibrate the standard input shaper algorith of Klipper using the other macros! This test should not be used before as it would be useless and the results invalid.


## Usage

Call the `VIBRATIONS_CALIBRATION` macro with the direction and speed range you want to measure. Here are the parameters available:

| parameters | default value | description |
|-----------:|---------------|-------------|
|SIZE|60|size in mm of the area where the movements are done|
|DIRECTION|"XY"|direction vector where you want to do the measurements. Can be set to either "XY", "AB", "ABXY", "A", "B", "X", "Y", "Z", "E"|
|Z_HEIGHT|20|z height to put the toolhead before starting the movements. Be careful, if your ADXL is under the nozzle, increase it to avoid a crash of the ADXL on the bed of the machine|
|VERBOSE|1|Wether to log the current speed in the console|
|MIN_SPEED|20|minimum speed of the toolhead in mm/s for the movements|
|MAX_SPEED|200|maximum speed of the toolhead in mm/s for the movements|
|SPEED_INCREMENT|2|speed increments of the toolhead in mm/s between every movements|
|TRAVEL_SPEED|200|speed in mm/s used for all the travels moves|
|ACCEL_CHIP|"adxl345"|accelerometer chip name in the config|


## Graphs description

## Analysis of the results

TODO: add the analysis part here
