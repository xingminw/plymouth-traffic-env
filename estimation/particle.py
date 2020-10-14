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

        # load demand and turning ratio
        demand_dict, turning_dict = self._load_demand_and_turning_ratio()

        # add additional class of segment and pipeline
        self._add_link_properties(demand_dict, turning_dict)

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

    def _add_link_properties(self, demand_dict, turning_dict):
        signalized_junction_list = list(self.signals.keys())
        for link_id in self.links.keys():
            link = self.links[link_id]
            segments = link.segments
            pipelines = link.pipelines

            # convert the link pipeline to a class
            new_pipeline_dict = {}
            for pip_idx in pipelines.keys():
                lane_list = pipelines[pip_idx]
                pipeline = PipeLine(lane_list[-1], lane_list)
                new_pipeline_dict[pipeline.id] = pipeline

            # add an additional property to the link
            self.links[link_id].link_type = "internal"

            if not (link.upstream_junction in signalized_junction_list):
                self.links[link_id].link_type = "source"
                start_edge = link.edge_list[0]

                if not (start_edge in demand_dict.keys()):
                    # print("There is something strange here,", link_id, "is wrongly recognized as source link")
                    self.links[link_id].link_type = "wrong"
                    continue            # skip this wrong link!!!
                else:
                    self.links[link_id].demand = demand_dict[start_edge]

            if not (link.downstream_junction in signalized_junction_list):
                self.links[link_id].link_type = "sink"
                continue

            # unless the sink is sink link, we associate it with turning ratio
            end_edge = link.edge_list[-1]
            edge = self.edges[end_edge]

            count_dict = {}
            for lane_id in edge.lanes_list:
                lane = self.lanes[lane_id]
                for downstream_lane in lane.downstream_lanes.keys():
                    downstream_edge = self.lanes[downstream_lane].edge_id
                    if not (downstream_edge in count_dict.keys()):
                        count_dict[downstream_edge] = 0
                    count_dict[downstream_edge] += 1

            for lane_id in edge.lanes_list:
                lane = self.lanes[lane_id]
                for downstream_lane in lane.downstream_lanes.keys():
                    downstream_info = lane.downstream_lanes[downstream_lane]
                    print(downstream_info)
                    downstream_edge = self.lanes[downstream_lane].edge_id
                    probability = turning_dict[end_edge][downstream_edge]

                    self.lanes[lane_id].downstream_lanes[downstream_lane]["prob"] = probability
                    pipeline_id = downstream_info["upstream_lane"]
                    if not (pipeline_id in new_pipeline_dict.keys()):
                        exit("There is something wrong loading the turning ratio to the pipeline...")

                    if new_pipeline_dict[pipeline_id].ratio is None:
                        new_pipeline_dict[pipeline_id].ratio = 0
                    new_pipeline_dict[pipeline_id].ratio += probability / count_dict[downstream_edge]

                    # dump the other information: direction, traffic light, link, etc...
                    new_pipeline_dict[pipeline_id].direction = downstream_info["dir"]
                    new_pipeline_dict[pipeline_id].signal = downstream_info["tl"]
                    new_pipeline_dict[pipeline_id].movement = downstream_info["link_id"]
            # load the new pipeline object to the link
            self.links[link_id].pipelines = new_pipeline_dict

    @staticmethod
    def _load_demand_and_turning_ratio():
        demand_file = env_config.network_flow_file.split("./")[-1]
        demand_tree = ET.ElementTree(file=demand_file)
        root = demand_tree.getroot()

        demand_dict = {}
        for subroot in root:
            origin_edge = subroot.attrib["from"]
            demand_level = float(subroot.attrib["number"]) / 3600.0
            demand_dict[origin_edge] = demand_level

        turning_file = env_config.turning_ratio_file.split("./")[-1]
        turning_tree = ET.ElementTree(file=turning_file)
        root = turning_tree.getroot()

        turning_dict = {}
        for subroot in root:
            if subroot.tag == "sink":
                continue
            for subsubroot in subroot:
                upstream_edge = subsubroot.attrib["id"]
                for _root in subsubroot:
                    downstream_edge = _root.attrib["id"]
                    probability = _root.attrib["probability"]

                    if not (upstream_edge in turning_dict.keys()):
                        turning_dict[upstream_edge] = {}
                    turning_dict[upstream_edge][downstream_edge] = float(probability)
                    break
        return demand_dict, turning_dict


class PipeLine(object):
    """
    pipeline is for each continuous lane
    """
    def __init__(self, pipeline_id, lane_list=None, ratio=None):
        self.id = pipeline_id
        self.lane_list = lane_list
        self.ratio = ratio

        self.direction = None
        self.signal = None
        self.movement = None
