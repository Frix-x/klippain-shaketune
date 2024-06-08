# Axis measurements

The `EXCITATE_AXIS_AT_FREQ` macro is particularly useful for troubleshooting mechanical vibrations or resonance issues. This macro allows you to maintain a specific excitation frequency for a set duration, enabling hands-on diagnostics. By touching different components during the excitation, you can identify the source of the vibration, as contact usually stops it.


## Usage

Here are the parameters available:

| parameters | default value | description |
|-----------:|---------------|-------------|
|FREQUENCY|25|excitation frequency (in Hz) that you want to maintain. Usually, it's the frequency of a peak on one of the graphs|
|DURATION|10|duration in second to maintain this excitation|
|ACCEL_PER_HZ|None|accel per Hz value used for the test. If unset, it will use the value from your `[resonance_tester]` config section (75 is the default)|
|AXIS|x|axis you want to excitate. Can be set to either "x", "y", "a", "b"|
|TRAVEL_SPEED|120|speed in mm/s used for all the travel movements (to go to the start position prior to the test)|
|Z_HEIGHT|None|Z height wanted for the test. This value can be used if needed to override the Z value of the probe_point set in your `[resonance_tester]` config section|

## Optional graphs description

![](../images/sf.png)

## Analysis of the results

