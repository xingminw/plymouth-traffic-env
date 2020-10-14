"""
inherit the whole class of traffic env and add a particle filter estimation part
"""

from abc import ABC
import xml.etree.ElementTree as ET
from traffic_envs.traffic_env import SignalizedNetwork

import matplotlib.pyplot as plt
import traffic_envs.config as env_config


class EstimatedNetwork(SignalizedNetwork, ABC):
    def __init__(self):
        SignalizedNetwork.__init__(self)

        # add additional class of segment and pipeline
        self._add_segment_pipeline()

        # load demand and turning ratio
        self._load_demand_and_turning_ratio()

    def _load_system_state(self):
        """
        override this function to add the particle filter
        :return:
        """
        SignalizedNetwork._load_system_state(self)
        self.particle_filter()

    def particle_filter(self):
        pass

    def _generate_observation(self):
        """

        :return:
        """
        pass

    def _add_segment_pipeline(self):
        signalized_junction_list = list(self.signals.keys())
        for link_id in self.links.keys():
            link = self.links[link_id]
            segments = link.segments
            pipelines = link.pipelines

            # add an additional property to the link
            self.links[link_id].link_type = "internal"

            if not (link.upstream_junction in signalized_junction_list):
                self.links[link_id].link_type = "source"

            if not (link.downstream_junction in signalized_junction_list):
                self.links[link_id].link_type = "sink"

    def _load_demand_and_turning_ratio(self):
        demand_file = env_config.network_flow_file.split("./")[-1]
        demand_tree = ET.ElementTree(demand_file)
        print(demand_tree)


class Segments(object):
    """
    minimum unit of the road segment
    """
    def __init__(self):
        pass


class Connector(object):
    """
    connector including the starting point and connector
        between segments as well as the intersection node
    """
    def __init__(self):
        pass


class PipeLine(object):
    """
    pipeline is for each continuous lane
    """
    def __init__(self):
        pass
