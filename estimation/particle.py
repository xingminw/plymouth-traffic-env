"""
inherit the whole class of traffic env and add a particle filter estimation part
"""

from abc import ABC
from traffic_envs.traffic_env import SignalizedNetwork


class EstimatedNetwork(SignalizedNetwork, ABC):
    def __init__(self):
        SignalizedNetwork.__init__(self)

        # add additional class of segment and pipeline
        self._add_segment_pipeline()

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
        pass


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

