# Shake&Tune documentation

![](./banner_long.png)


When perfecting 3D prints and tuning your printer, there is all that resonance testing stuff that Shake&Tune will try to help you with. But keep in mind that it's part of a complete process, and Shake&Tune alone won't magically make your printer print at lightning speed. Also, when using the tools, **it's important to get back to the original need: good prints**.

While there are some ideal goals described in this documentation, you need to understand that it's not always possible to achieve them due to a variety of factors unique to each printer, such as assembly precision, components quality and brand, components wear, etc. Even a different accelerometer can give different results. But that's not a problem; the primary goal is to produce clean and satisfactory prints. If your test prints look good and meet your standards, even if the response curves aren't perfect, you're on the right track. **Trust your printer and your print results more than chasing ideal graphs!** If it's satisfactory, there's no need for further adjustments.

First, you may want to read the **[input shaping and tuning generalities](./is_tuning_generalities.md)** documentation to understand how it all works and what to look for when taking these measurements.


## Shake&Tune macros

| Shake&Tune command | Resulting graphs example |
|:------|:-------:|
|[`AXES_MAP_CALIBRATION`](./macros/axes_map_calibration.md)<br /><br />Verify that your accelerometer is working correctly and automatically find its Klipper's `axes_map` parameter | [<img src="./images/axesmap_example.png">](./macros/axes_map_calibration.md) |
|[`COMPARE_BELTS_RESPONSES`](./macros/compare_belts_responses.md)<br /><br />Generate a differential belt resonance graph to verify relative belt tensions and belt path behaviors on a CoreXY or CoreXZ printer | [<img src="./images/belts_example.png">](./macros/compare_belts_responses.md) |
|[`AXES_SHAPER_CALIBRATION`](./macros/axes_shaper_calibrations.md)<br /><br />Create the usual input shaper graphs to tune Klipper's input shaper filters and reduce ringing/ghosting | [<img src="./images/axis_example.png">](./macros/axes_shaper_calibrations.md) |
|[`CREATE_VIBRATIONS_PROFILE`](./macros/create_vibrations_profile.md)<br /><br />Measure your global machine vibrations as a function of toolhead direction and speed to find problematic ranges where the printer could be exposed to more VFAs in order to optimize your slicer speed profiles and TMC drivers parameters | [<img src="./images/vibrations_example.png">](./macros/create_vibrations_profile.md) |
|[`EXCITATE_AXIS_AT_FREQ`](./macros/excitate_axis_at_freq.md)<br /><br />Maintain a specific excitation frequency, useful to inspect parasite peaks and find out what is resonating | [<img src="./images/excitate_at_freq_example.png">](./macros/excitate_axis_at_freq.md) |


## Resonance testing workflow

A standard tuning workflow might look something like this:

```mermaid
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'lineColor': '#232323',
      'primaryTextColor': '#F2055C',
      'secondaryColor': '#D3D3D3',
      'tertiaryColor': '#FFFFFF'
    }
  }
}%%

flowchart TB
    subgraph Tuning Workflow
    direction LR
    start([Start]) --> tensionBelts[Tension your\nbelts as best\n as possible]
    checkmotion --> tensionBelts
    tensionBelts --> SnT_Belts[Run Shake&Tune\nbelts comparison tool]
    SnT_Belts --> goodbelts{Check the documentation\nDoes belts comparison profiles\nlook decent?}
    goodbelts --> |YES| SnT_IS[Run Shake&Tune\naxis input shaper tool]
    goodbelts --> |NO| checkmotion[Fix your mechanical assembly\nand your motion system]
    SnT_IS --> goodIS{Check the documentation\nDoes axis profiles and\n input shapers look decent?}
    goodIS --> |YES| SnT_Vibrations[Run Shake&Tune\nvibration profile tool]
    goodIS--> |NO| checkmotion
    SnT_Vibrations --> goodvibs{Check the documentation\nAre the graphs OK?\nSet the speeds in\nyour slicer profile}
    goodvibs --> |YES| pressureAdvance[Tune your\npressure advance]
    goodvibs --> |NO| checkTMC[Dig into TMC drivers\ntuning if you want to]
    goodvibs --> |NO| checkmotion
    checkTMC --> SnT_Vibrations
    pressureAdvance --> extrusionMultiplier[Tune your\nextrusion multiplier]
    extrusionMultiplier --> testPrint[Do a test print]
    testPrint --> printGood{Is the print good?}
    printGood --> |YES| unicorn{want to chase unicorns}
    printGood --> |NO -> Underextrusion / Overextrusion| extrusionMultiplier
    printGood --> |NO -> Corner humps and no ghosting| pressureAdvance
    printGood --> |NO -> Visible VFAs| SnT_Vibrations
    printGood --> |NO -> Ghosting, ringing, resonance| SnT_IS
    unicorn --> |NO| done
    unicorn --> |YES| SnT_Belts
    end

    classDef standard fill:#70088C,stroke:#150140,stroke-width:4px,color:#ffffff;
    classDef questions fill:#FF8D32,stroke:#F24130,stroke-width:4px,color:#ffffff;
    classDef startstop fill:#F2055C,stroke:#150140,stroke-width:3px,color:#ffffff;
    class start,done startstop;
    class goodbelts,goodIS,goodvibs,printGood,unicorn questions;
    class tensionBelts,checkmotion,SnT_Belts,SnT_IS,SnT_Vibrations,pressureAdvance,extrusionMultiplier,testPrint,checkTMC standard;
```


## Complementary ressources

  - [Sineos post](https://klipper.discourse.group/t/interpreting-the-input-shaper-graphs/9879) in the Klipper knowledge base
