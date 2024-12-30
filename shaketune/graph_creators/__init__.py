# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: __init__.py
# Description: Imports various graph creator classes for the Shake&Tune package.

import os
import sys


def get_shaper_calibrate_module():
    if os.environ.get('SHAKETUNE_IN_CLI') != '1':
        from ... import shaper_calibrate, shaper_defs
    else:
        shaper_calibrate = sys.modules['shaper_calibrate']
        shaper_defs = sys.modules['shaper_defs']
    return shaper_calibrate.ShaperCalibrate(printer=None), shaper_defs


from .axes_map_graph_creator import AxesMapGraphCreator as AxesMapGraphCreator  # noqa: E402
from .belts_graph_creator import BeltsGraphCreator as BeltsGraphCreator  # noqa: E402
from .graph_creator import GraphCreator as GraphCreator  # noqa: E402
from .graph_creator_factory import GraphCreatorFactory as GraphCreatorFactory  # noqa: E402
from .shaper_graph_creator import ShaperGraphCreator as ShaperGraphCreator  # noqa: E402
from .static_graph_creator import StaticGraphCreator as StaticGraphCreator  # noqa: E402
from .vibrations_graph_creator import VibrationsGraphCreator as VibrationsGraphCreator  # noqa: E402
