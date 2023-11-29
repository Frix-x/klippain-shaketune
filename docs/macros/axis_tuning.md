# Axis measurements

The `AXES_SHAPER_CALIBRATION` macro is used to measure and plot the axis behavior in order to tune Klipper's input shaper system.


## Usage

**Before starting, ensure that the belts are properly tensioned** and that you already have good and clear belt graphs (see [the previous section](./belts_tuning.md)).

Then, call the `AXES_SHAPER_CALIBRATION` macro and look for the graphs in the results folder. Here are the parameters available:

| parameters | default value | description |
|-----------:|---------------|-------------|
|VERBOSE|1|Whether or not to log things in the console|
|FREQ_START|5|Starting excitation frequency|
|FREQ_END|133|Maximum excitation frequency|
|HZ_PER_SEC|1|Number of Hz per seconds for the test|
|AXIS|"all"|Axis you want to test in the list of "all", "X" or "Y"|


## Graphs description

![](../images/shaper_graphs/shaper_graph_explanation.png)

## Analysis of the results

### Generalities

To effectively analyze input shaper graphs, there is no one-size-fits-all approach due to the variety of factors that can impact the 3D printer's performance or input shaper measurements. However, here are some hints on reading the graphs:
  - A graph with a **single and thin peak** well detached from the background noise is ideal, as it can be easily filtered by input shaping. But depending on the machine and its mechanical configuration, it's not always possible to obtain this shape. The key to getting better graphs is a clean mechanical assembly with a special focus on the rigidity and stiffness of everything, from the table the printer sits on to the frame and the toolhead.
  - As for the belt graphs, **focus on the shape of the graphs, not the values**. Indeed, the energy value doesn't provide much useful information. Use it only to compare two of your own graphs and to measure the impact of your mechanical changes between two consecutive tests, but never use it to compare against graphs from other people or other machines.

![](../images/shaper_graphs/shaper_recommandations.png)

For setting your Input Shaping filters, rely on the auto-computed values displayed in the top right corner of the graph. Here's a breakdown of the legend for a better grasp:
  - **Filtering algortihms**: Klipper automatically computes these lines. This computation works pretty well if the graphs are clean enough. But if your graphs are junk, it can't do magic and will give you pretty bad recommendations. It's better to address the mechanical issues first before continuing. Each shapers has its pro and cons:
    * `ZV` is a pretty light filter and usually has some remaining vibrations. My recommendation would be to use it only if you want to do speed benchies and get the highest acceleration values while maintaining a low amount of smoothing on your parts. If you have "perfect" graphs and do not care that much about some remaining ringing, you can try it. 
    * `MZV` is usually the top pick for well-adjusted machines. It's a good compromise for low remaining vibrations while still allowing pretty good acceleration values. Keep in mind, `MZV` is only recommended by Klipper on good graphs.
    * `EI` can be used as a fallback for challenging graphs. But first, try to fix your mechanical issues before using it: almost every printer should be able to run `MZV` instead.
    * `2HUMP_EI` and `3HUMP_EI` are last-resort choices. Usually, they lead to a high level of smoothing in order to suppress the ringing while also using relatively low acceleration values. If they pop up as suggestions, it's likely your machine has underlying mechanical issues (that lead to pretty bad or "wide" graphs).
  - **Recommended Acceleration** (`accel<=...`): This isn't a standalone figure. It's essential to also consider the `vibr` and `sm` values as it's a compromise between the three. They will give you the percentage of remaining vibrations and the smoothing after Input Shaping, when using the recommended acceleration. Nothing will prevent you from using higher acceleration values; they are not a limit. However, when doing so, Input Shaping may not be able to suppress all the ringing on your parts. Finally, keep in mind that high acceleration values are not useful at all if there is still a high level of remaining vibrations: you should address any mechanical issues first.
  - **The remaining vibrations** (`vibr`): This directly correlates with ringing. It correspond to the total value of the blue "after shaper" signal. Ideally, you want a filter with minimal or zero vibrations.
  - **Shaper recommendations**: This script will give you some tailored recommendations based on your graphs. Pick the one that suit your needs:
    * The "performance" shaper is Klipper's original suggestion that is good for high acceleration while also sometimes allowing a little bit of remaining vibrations. Use it if your goal is speed printing and you don't care much about some remaining ringing.
    * The "low vibration" shaper aims for the lowest level of remaining vibration to ensure the best print quality with minimal ringing. This should be the best bet for most users.
    * Sometimes, only a single recommendation called "best" shaper is presented. This means that either no suitable "low vibration" shaper was found (due to a high level of vibration or with too much smoothing) or because the "performance" shaper is also the one with the lowest vibration level.
  - **Damping Ratio**: Displayed at the end, this estimatation is only reliable when the graph shows a distinct, standalone and clean peak. On a well tuned machine, setting the damping ratio (instead of Klipper's 0.1 default value) can further reduce the ringing at high accelerations and with higher square corner velocities.

Then, add to your configuration:
```
[input_shaper]
shaper_freq_x: ... # center frequency for the X axis filter
shaper_type_x: ... # filter type for the X axis
shaper_freq_y: ... # center frequency for the Y axis filter
shaper_type_y: ... # filter type for the Y axis
damping_ratio_x: ... # damping ratio for the X axis
damping_ratio_y: ... # damping ratio for the Y axis
```

### Useful facts and myths debunking

Some people suggest to cap data at 100 Hz by manually editing the .csv file, thinking values beyond that are wrong. But this can be misleading. The excitation and system's response frequencies differ, and aren't directly linked. You might see vibrations beyond the excitation range, and removing them from the file just hides potential issues. Though these high-frequency vibrations might not always affect print quality, they could signal mechanical problems. Instead of hiding them, look into resolving these issues.

Regarding printer components, I do not recommend using an extra-light X-beam (aluminum or carbon). They might seem ideal due to their weight, but there's more to consider than just mass such as the rigidity (see the [theory behind it](../is_tuning_generalities.md#theory-behind-it)). These light beams can be more flexible and will impact negatively the Y axis graphs as they will flex under high accelerations.

Finally, keep in mind that each axis has its own properties, such as mass and geometry, which will lead to different behaviors for each of them and will require different filters. Using the same input shaping settings for both axes is only valid if both axes are similar mechanically: this may be true for some machines, mainly Cross gantry configurations such as [CroXY](https://github.com/CroXY3D/CroXY) or [Annex-Engineering](https://github.com/Annex-Engineering) printers, but not for others.


## Examples of graphs

In this section, I'll walk you through some random graphs sourced online or shared with me for analysis. My aim is to highlight the good and the not-so-good, offering insights to help you refine your printer's Input Shaping settings.

That said, interpreting Input Shaper graphs isn't an exact science. While we can make educated guesses and highlight potential issues from these graphs, pinpointing exact causes isn't always feasible. So, consider the upcoming graphs and their comments as pointers on your input shaping journey, rather than hard truths.

### Good graphs

These two graphs are considered good and is what you're aiming for. They each display a single, distinct peak that stands out clearly against the background noise. Note that the main frequencies of the X and Y graph peaks differ. This variance is expected and normal, as explained in the last point of the [useful facts and myths debunking](#useful-facts-and-myths-debunking) section.

| Good X graph | Good Y graph |
| --- | --- |
| ![](../images/shaper_graphs/low_canbus_solved.png) | ![](../images/shaper_graphs/reso_good_y.png) |

### Low frequency energy

These graphs have some low frequency energy (signal near 0 Hz) on a rather low maximum amplitude (around 1e2 or 1e3). This means that there is some binding, rubbing or grinding during movements: basically, something isn't moving freely. Minor low frequency energy in the graphs might be due to a lot of issues such as a faulty idler/bearing or an overly tightened carriage screw that prevent it to move freely on its linear rail, ... However, major low frequency energy suggest more important problems like improper belt routing (the most common), or defective motor, ...

Here's how to troubleshoot the issue:
  1. **Belts Examination**:
     - Ensure your belts are properly routed.
     - Check for correct alignment of the belts on all bearing flanges during movement (check them during a print).
     - Belt dust is often a sign of misalignment or wear.
  2. **Toolhead behavior on CoreXY printers**: With motors off and the toolhead centered, gently push the Y-axis front-to-back. The toolhead shouldn't move left or right. If it does, one of the belts might be obstructed and requires inspection to find out the problem.
  3. **Gantry Squareness**:
     - Ensure your gantry is perfectly parallel and square. You can refer to [Nero3D's de-racking video](https://youtu.be/cOn6u9kXvy0?si=ZCSdWU6br3Y9rGsy) for guidance.
     - After removing the belts, test the toolhead's movement by hand across all positions. Movement should be smooth with no hard points or areas of resistance.

| Small binding | Heavy binding |
| --- | --- |
| ![](../images/shaper_graphs/binding.png) | ![](../images/shaper_graphs/binding2.png) |

### Double peaks or wide peaks

Such graph patterns can arise from various factors, and there isn't a one-size-fits-all solution. To address them:
  1. A wobbly table can be the cause. So first thing to do is to try with the printer directly on the floor.
  2. Ensure optimal belt tension using the [`BELTS_SHAPER_CALIBRATION` macro](./belts_tuning.md).
  3. If problems persist, it might be due to an improperly squared gantry. For correction, refer to [Nero3D's de-racking video](https://youtu.be/cOn6u9kXvy0?si=ZCSdWU6br3Y9rGsy).
  4. If it's still there... you will need to find out what is resonating to fix it. You can use the `EXCITATE_AXIS_AT_FREQ` macro to help you find it.

| Two peaks | Single wide peak |
| --- | --- |
| ![](../images/shaper_graphs/bad_racking.png) | ![](../images/shaper_graphs/bad_racking2.png) |

### Problematic CANBUS speed

Using CANBUS toolheads with an integrated ADXL chip can sometimes pose challenges if the CANBUS speed is set too low. While users might lower the bus speed to fix Klipper's timing errors, this change will also affect input shaping measurements. An example outcome of a low bus speed is the following graph that, though generally well-shaped, appears jagged and spiky throughout. Additional low-frequency energy might also be present. For optimal ADXL board operation on your CANBUS toolhead, a speed setting of 500k is the minimum, but 1M is advisable.

| CANBUS problem present | CANBUS problem solved |
| --- | --- |
| ![](../images/shaper_graphs/low_canbus.png) | ![](../images/shaper_graphs/low_canbus_solved.png) |

### Toolhead or TAP wobble

The [Voron TAP](https://github.com/VoronDesign/Voron-Tap) can introduce anomalies to input shaper graphs, notably on the X graph. Its design with an internal MGN rail introduces a separate and decoupled mass, leading to its own resonance, typically around 125Hz. Combatting this can be pretty challenging, but using premium components and a careful assembly can help mitigate the issue. Ensure you employ a good quality and well-preloaded TAP MGN rail for optimal assembly stiffness, coupled with genuine and strong N52 magnets (avoid lower-quality N35 or N45 substitutes often found on chinese marketplaces). Prioritize careful assembly and consider using the TAP Rev8 version or above.

Additionally, without a Voron TAP, small 125hz peaks can sometimes tie back to the toolhead itself. Common culprits include loosely fitted screws or a bad quality X linear MGN axis that can have some play in the carriage, leading to slight toolhead wobbling. This is often represented as a Z component in the graphs.

If your graph shows this kind of anomalies, begin by disassembling the toolhead up to the X carriage. Check for any looseness, then reassemble, ensuring everything is tightened properly for a rigid assembly. Also, don't forget to check your extruder and validate its assembly as well. Finally, ensure you have some filament loaded during measurements to prevent extruder gear vibrations.

| TAP wobble problem | TAP wobble problem partially mitigated<br/>Or toolhead wobbling |
| --- | --- |
| ![](../images/shaper_graphs/TAP_125hz.png) | ![](../images/shaper_graphs/TAP_125hz_2.png) |

### Unbalanced fan

The presence of an unbalanced or badly running fan can be directly observed in the graphs. While you should let the toolhead fans off during the final IS tuning, you can use this test to validate their correct behavior: an unbalanced fan usually add some very thin peak around 100-150Hz that disapear when the fan is off. Also please note that an unbalanced fan constant frequency is manifested as a vertical line on the bottom spectrogram.

| Unbalanced fan running | Unbalanced fan off |
| --- | --- |
| ![](../images/shaper_graphs/unbalanced_fan_on.png) | ![](../images/shaper_graphs/unbalanced_fan_off.png) |

### Crazy graphs and miscs

The depicted graphs are challenging to analyze due to the overwhelming noise across the spectrum. Such patterns are often associated with an improperly assembled and non-squared mechanical structure. To address this:
  1. Refer to the [Low frequency energy](#low-frequency-energy) section for troubleshooting steps.
  2. If unresolved, consider disassembling the entire gantry, inspect the printed and mechanical components, and ensure meticulous reassembly. A thorough and careful assembly should help alleviate the issue. Measure again post-assembly for changes.

Also please note that for this kind of graphs, as they are mainly consisting of noise, Klipper's algorithm recommendations must not be used and will not help with ringing. You will need to fix your mechanical issues instead!

| Crazy X | Crazy Y |
| --- | --- |
| ![](../images/shaper_graphs/chaos_x.png) | ![](../images/shaper_graphs/chaos_y.png) |
