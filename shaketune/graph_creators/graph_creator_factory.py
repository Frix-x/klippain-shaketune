# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: graph_creator_factory.py
# Description: Contains a factory class to create the different graph creators.


from ..shaketune_config import ShakeTuneConfig
from .graph_creator import GraphCreator


class GraphCreatorFactory:
    @staticmethod
    def create_graph_creator(graph_type: str, config: ShakeTuneConfig) -> GraphCreator:
        creator_class = GraphCreator.registry.get(graph_type, None)
        if not creator_class:
            raise NotImplementedError(f'Graph creator for {graph_type} not implemented!')
        return creator_class(config)
