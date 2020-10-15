"""
inherit the whole class of traffic env and add a particle filter estimation part
"""

from abc import ABC
from traffic_envs.traffic_env import SignalizedNetwork

import numpy as np
# import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
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

    def set_mode(self, actuate_control):
        self.actuate_control = actuate_control
        self._load_network_topology()

        # load demand and turning ratio
        demand_dict, turning_dict = self._load_demand_and_turning_ratio()

        # add additional class of segment and pipeline
        self._add_link_properties(demand_dict, turning_dict)

    def particle_filter(self):
        self._sort_vehicle_within_link()

        for link_id in self.links.keys():
            link = self.links[link_id]
            if link.link_type == "source":
                pipelines = link.pipelines
                for pip_id in pipelines.keys():
                    pipeline = pipelines[pip_id]
                    pr_vehicles = pipeline.previous_vehicles
                    vehicles = pipeline.vehicles


                    new_arrival = False
                    # event: new arrival
                    if len(vehicles) > 0:
                        if not (vehicles[0] in pr_vehicles):
                            new_arrival = True
                continue
            elif link.link_type == "sink":
                pass
            elif link.link_type == "internal":
                pass
            else:
                pass

    def _get_cv_location(self, vehicle_list):
        cv_list = []
        distance_list = []
        for vehicle_id in vehicle_list:
            vehicle = self.vehicles[vehicle_id]
            if vehicle.cv_type:
                cv_list.append(vehicle_id)
                distance_list.append(vehicle.link_pos_list[-1])
        return cv_list, distance_list

    def _generate_observation(self):
        """

        :return:
        """
        pass

    def _add_link_properties(self, demand_dict, turning_dict):
        signalized_junction_list = list(self.signals.keys())
        for link_id in self.links.keys():
            link = self.links[link_id]
            pipelines = link.pipelines
            self.links[link_id].ingress_lanes = len(self.edges[link.edge_list[0]].lanes_list)

            # convert the link pipeline to a class
            new_pipeline_dict = {}
            for pip_idx in pipelines.keys():
                lane_list = pipelines[pip_idx]
                pipeline = PipeLine(lane_list[-1], lane_list)
                new_pipeline_dict[pipeline.id] = pipeline

            # load the new pipeline object to the link
            self.links[link_id].pipelines = new_pipeline_dict

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
            # initiate the turning ratio with 0
            self.links[link_id].left_turn = 0
            self.links[link_id].right_turn = 0
            self.links[link_id].through = 0

            end_edge = link.edge_list[-1]
            edge = self.edges[end_edge]

            for lane_id in edge.lanes_list:
                lane = self.lanes[lane_id]
                for downstream_lane in lane.downstream_lanes.keys():
                    downstream_info = lane.downstream_lanes[downstream_lane]
                    downstream_edge = self.lanes[downstream_lane].edge_id
                    probability = turning_dict[end_edge][downstream_edge]

                    self.lanes[lane_id].downstream_lanes[downstream_lane]["prob"] = probability
                    pipeline_id = downstream_info["upstream_lane"]
                    if not (pipeline_id in new_pipeline_dict.keys()):
                        exit("There is something wrong loading the turning ratio to the pipeline...")

                    if downstream_info["dir"] == "s":
                        self.links[link_id].through = probability
                    elif downstream_info["dir"] == "l" or downstream_info["dir"] == "L":
                        self.links[link_id].left_turn = probability
                    elif downstream_info["dir"] == "r":
                        self.links[link_id].right_turn = probability
                    else:
                        exit("Direction " + str(downstream_info["dir"]) + " not recognized!!")

                    # dump the other information: direction, traffic light, link, etc...
                    if new_pipeline_dict[pipeline_id].direction is None:
                        new_pipeline_dict[pipeline_id].direction = ""

                    new_pipeline_dict[pipeline_id].direction += downstream_info["dir"]
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

    def _sort_vehicle_within_link(self):
        """
        sort the vehicle within the link
            split into different pipeline
        :return:
        """
        link_vehicle_dict = self.link_vehicle_dict
        for link_id in self.links.keys():
            link = self.links[link_id]
            link.lane_change_events = {}

            # get the pipelines and segments of the link
            pipelines = link.pipelines
            segments = link.segments

            if not (link_id in link_vehicle_dict.keys()):
                continue
            vehicle_list = link_vehicle_dict[link_id]["vehicles"]

            pipeline_vehicle_dict = {}
            for vehicle_id in vehicle_list:
                vehicle = self.vehicles[vehicle_id]
                vehicle_lane = vehicle.lane_list[-1]
                vehicle_edge = self.lanes[vehicle_lane].edge_id
                vehicle_pos = vehicle.link_pos_list[-1]

                # get the segment
                segment_assigned = False
                for segment_id in segments.keys():
                    segment_edges = segments[segment_id]
                    if vehicle_edge in segment_edges:
                        self.vehicles[vehicle_id].segment_list.append(segment_id)
                        segment_assigned = True
                        break
                if not segment_assigned:
                    original_segment = self.vehicles[vehicle_id].segment_list[-1]
                    self.vehicles[vehicle_id].segment_list.append(original_segment)

                # get the pipeline id
                pipeline_assigned = False
                for pipeline_id in pipelines:
                    pipeline_lanes = pipelines[pipeline_id].lane_list
                    if vehicle_lane in pipeline_lanes:
                        self.vehicles[vehicle_id].pipeline_list.append(pipeline_id)
                        pipeline_assigned = True

                        if not (pipeline_id in pipeline_vehicle_dict.keys()):
                            pipeline_vehicle_dict[pipeline_id] = {"vehicles": [], "dis": []}
                        pipeline_vehicle_dict[pipeline_id]["vehicles"].append(vehicle_id)
                        pipeline_vehicle_dict[pipeline_id]["dis"].append(vehicle_pos)
                if not pipeline_assigned:
                    original_pipeline = self.vehicles[vehicle_id].pipeline_list[-1]
                    self.vehicles[vehicle_id].pipeline_list.append(original_pipeline)
                    if not (original_pipeline in pipeline_vehicle_dict.keys()):
                        pipeline_vehicle_dict[original_pipeline] = {"vehicles": [], "dis": []}
                    pipeline_vehicle_dict[original_pipeline]["vehicles"].append(vehicle_id)
                    pipeline_vehicle_dict[original_pipeline]["dis"].append(vehicle_pos)

                # detect the lane-changing event
                if len(vehicle.link_list) > 2:
                    if len(vehicle.pipeline_list) > 2:
                        if vehicle.link_list[-1] == vehicle.link_list[-2]:                # remain in the same link
                            if vehicle.pipeline_list[-1] != vehicle.pipeline_list[-2]:    # different pipeline
                                self.links[link_id].lane_change_events[vehicle_id] = \
                                    [[vehicle.pipeline_list[-2], vehicle.pipeline_list[-1]]]

            # sort the vehicle within different pipelines
            for pipeline_id in pipeline_vehicle_dict.keys():
                distance_list = pipeline_vehicle_dict[pipeline_id]["dis"]
                vehicle_id_list = pipeline_vehicle_dict[pipeline_id]["vehicles"]
                sequence = np.argsort(distance_list)
                new_vehicle_list = []
                new_distance_list = []
                for sdx in sequence:
                    new_vehicle_list.append(vehicle_id_list[sdx])
                    new_distance_list.append(distance_list[sdx])

                new_distance_list = [0] + new_distance_list + [link.length]

                # dump the current vehicle to the previous vehicle
                self.links[link_id].pipelines[pipeline_id].previous_vehicles = \
                    self.links[link_id].pipelines[pipeline_id].vehicles

                # update the vehicle in the link pipelines
                self.links[link_id].pipelines[pipeline_id].vehicles = new_vehicle_list
                new_vehicle_list = [None] + new_vehicle_list + [None]

                for vdx in range(len(new_vehicle_list) - 2):
                    following_vehicle = new_vehicle_list[vdx]
                    current_vehicle = new_vehicle_list[vdx + 1]
                    leading_vehicle = new_vehicle_list[vdx + 2]
                    following_dis = new_distance_list[vdx + 2] - new_distance_list[vdx + 1]
                    leading_dis = new_distance_list[vdx + 1] - new_distance_list[vdx]
                    self.vehicles[current_vehicle].leading_dis_list.append(leading_dis)
                    self.vehicles[current_vehicle].leading_vehicle_list.append(leading_vehicle)
                    self.vehicles[current_vehicle].following_dis_list.append(following_dis)
                    self.vehicles[current_vehicle].following_vehicle_list.append(following_vehicle)

            # LESSON: DEEPCOPY COULD BE VERY VERY SLOW !!!!!
            # find the leading cv and following cv
            for vehicle_id in vehicle_list:
                # vehicle = self.vehicles[vehicle_id]
                time_out_max = 1000                  # set a time out threshold,
                # forward search
                vehicle_cursor = vehicle_id
                count = 0
                while True:
                    count += 1
                    if count > time_out_max:
                        exit("Time out error for finding the forward cv")
                    leading_vehicle = self.vehicles[vehicle_cursor].leading_vehicle_list[-1]
                    if leading_vehicle is None:
                        self.vehicles[vehicle_id].leading_cv_list.append(leading_vehicle)
                        break
                    leading_type = self.vehicles[leading_vehicle].cv_type
                    if leading_type:
                        self.vehicles[vehicle_id].leading_cv_list.append(leading_vehicle)
                        break
                    vehicle_cursor = leading_vehicle

                # backward search
                vehicle_cursor = vehicle_id
                count = 0
                while True:
                    count += 1
                    if count > time_out_max:
                        exit("Time out error for finding the backward cv")
                    following_vehicle = self.vehicles[vehicle_cursor].following_vehicle_list[-1]
                    if following_vehicle is None:
                        self.vehicles[vehicle_id].following_cv_list.append(following_vehicle)
                        break
                    following_type = self.vehicles[following_vehicle].cv_type
                    if following_type:
                        self.vehicles[vehicle_id].following_cv_list.append(following_vehicle)
                        break
                    vehicle_cursor = following_vehicle


class PipeLine(object):
    """
    pipeline is for each continuous lane
    """
    def __init__(self, pipeline_id, lane_list=None):
        self.id = pipeline_id
        self.lane_list = lane_list

        self.direction = None
        self.signal = None
        self.movement = None
        self.downstream_links = None

        # real-time state
        self.vehicles = []
        self.previous_vehicles = []
