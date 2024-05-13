#!/usr/bin/env python3

from .axes_input_shaper import axes_shaper_calibration as axes_shaper_calibration
from .axes_map import axes_map_calibration as axes_map_calibration
from .belts_comparison import compare_belts_responses as compare_belts_responses
from .static_freq import excitate_axis_at_freq as excitate_axis_at_freq
from .vibrations_profile import create_vibrations_profile as create_vibrations_profile

AXIS_CONFIG = [
    {'axis': 'x', 'direction': (1, 0, 0), 'label': 'axis_X'},
    {'axis': 'y', 'direction': (0, 1, 0), 'label': 'axis_Y'},
    {'axis': 'a', 'direction': (1, -1, 0), 'label': 'belt_A'},
    {'axis': 'b', 'direction': (1, 1, 0), 'label': 'belt_B'},
]

# graph_creators = {
#     'axesmap': (AxesMapFinder, lambda gc: gc.configure(options.accel_used, options.chip_name)),
#     'belts': (BeltsGraphCreator, None),
#     'shaper': (ShaperGraphCreator, lambda gc: gc.configure(options.scv, options.max_smoothing)),
#     'vibrations': (
#         VibrationsGraphCreator,
#         lambda gc: gc.configure(options.kinematics, options.accel_used, options.chip_name, options.metadata),
#     ),
# }
