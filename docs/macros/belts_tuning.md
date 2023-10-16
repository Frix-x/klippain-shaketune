# Belt relative difference measurements

The `BELTS_SHAPER_CALIBRATION` macro is dedicated for CoreXY machines where it can help you to diagnose belt path problems by measuring and plotting the differences between their behavior. It will also help you tension your belts at the same tension.


## Usage

**Before starting, ensure that the belts are properly tensioned**. For example, you can follow the [Voron belt tensioning documentation](https://docs.vorondesign.com/tuning/secondary_printer_tuning.html#belt-tension). This is crucial!

Then, call the `BELTS_SHAPER_CALIBRATION` macro and look for the graphs in the results folder. Here are the parameters available:

| parameters | default value | description |
|-----------:|---------------|-------------|
|VERBOSE|1|Wether to log things in the console|
|FREQ_START|5|Starting excitation frequency|
|FREQ_END|133|Maximum excitation frequency|
|HZ_PER_SEC|1|Number of Hz per seconds for the test|


## Graphs description

## Analysis of the results

On these graphs, you want both curves to look similar and overlap to form a single curve. Try to make them fit as closely as possible. It's acceptable to have "noise" around the main peak, but it should be present on both curves with a comparable amplitude. Keep in mind that when you tighten a belt, its main peak should move diagonally toward the upper right corner, changing significantly in amplitude and slightly in frequency. Additionally, the magnitude order of the main peaks *should typically* range from ~100k to ~1M on most machines.

The resonant frequency/amplitude of the curves depends primarily on three parameters (and the actual tension):
  - the *mass of the toolhead*, which is identical for both belts and has no effect here
  - the *belt "elasticity"*, which changes over time as the belt wears. Ensure that you use the **same belt brand and type** for both A and B belts and that they were **installed at the same time**
  - the *belt path length*, which is why they must have the **exact same number of teeth** so that one belt path is not longer than the other when tightened at the same tension

**If these three parameters are met, there is no way that the curves could be different** or you can be sure that there is an underlying problem in at least one of the belt paths. Also, if the belt graphs have low amplitude curves (no distinct peaks) and a lot of noise, you will probably also have poor input shaper graphs. So before you continue, ensure that you have good belt graphs or fix your belt paths. Start by checking the belt tension, bearings, gantry screws, alignment of the belts on the idlers, and so on.


## Examples of graphs

| Comment | Belt graphs examples 1 | Belt graphs examples 2 |
| --- | --- | --- |
| **Both of these two graphs are considered good**. As you can see, the main peak doesn't have to be perfect if you can get both curves to overlap | ![](./images/IS_docs/belt_graphs/perfect%20graph.png) | ![](./images/resonances_belts_example.png) |
| **These two graphs show incorrect belt tension**: in each case, one of the belts has insufficient tension (first is B belt, second is A belt). Begin by tightening it half a turn and measuring again | ![](./images/IS_docs/belt_graphs/different_tensions.png) | ![](./images/IS_docs/belt_graphs/different_tensions2.png) |
| **These two graphs indicate a belt path problem**: the belt tension could be adequate, but something else is happening in the belt paths. Start by checking the bearings and belt wear, or belt alignment | ![](./images/IS_docs/belt_graphs/belts_problem.png) | ![](./images/IS_docs/belt_graphs/belts_problem2.png) |
