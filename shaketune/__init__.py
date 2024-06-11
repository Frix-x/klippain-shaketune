############################################
###### INPUT SHAPER KLIPPAIN WORKFLOW ######
############################################
# Written by Frix_x#0161 #

# This module functions as a plugin within Klipper, aimed at enhancing printer diagnostics. It serves multiple purposes:
# 1. Diagnosing and pinpointing vibration sources in the printer.
# 2. Conducting standard axis input shaper tests on the XY axes to determine the optimal input shaper filter.
# 3. Executing a specialized half-axis test for CoreXY printers to analyze and compare the frequency profiles of individual belts.


from .shaketune import ShakeTune as ShakeTune


def load_config(config) -> ShakeTune:
    return ShakeTune(config)
