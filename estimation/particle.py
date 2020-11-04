"""
inherit the whole class of traffic env and add a particle filter estimation part
"""

from abc import ABC
from scipy.stats import uniform
from copy import deepcopy
from estimation.car_following import SimplifiedModel
from traffic_envs.traffic_env import SignalizedNetwork, Link
from traffic_envs.utils import\
    generate_new_route_configuration, delete_buffer_file

import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import traffic_envs.config as env_config

__author__ = "Xingmin Wang"
LIBSUMO_FLAG = False


# import traci
sumoBinary = env_config.sumoBinary
if env_config.GUI_MODE:
    import traci
    sumoBinary += "-gui"
else:
    # use libsumo to replace traci to speed up (cannot use gui...)
    try:
        import libsumo as traci
        LIBSUMO_FLAG = True
    except ImportError:
        LIBSUMO_FLAG = False
        print("libsumo is not installed correctly, use traci instead...")
        import traci


class EstimatedNetwork(SignalizedNetwork, ABC):
    def __init__(self):
        SignalizedNetwork.__init__(self)

        # number of particles
        self.particle_number = 3

        # number of grid size
        self.grid_size = 14

        # load demand and turning ratio
        demand_dict, turning_dict = self._load_demand_and_turning_ratio()

        # add additional class of segment and pipeline
        self._add_link_properties(demand_dict, turning_dict)

    def _load_system_state(self):
        """
        override this function to add the particle filter
        :return:
        """
        print(self.time_step)
        SignalizedNetwork._load_system_state(self)
        self._sort_vehicle_within_link()
        # self._output_particle_posterior()
        self.particle_filter()

    def set_mode(self, actuate_control):
        self.actuate_control = actuate_control
        self._load_network_topology()

        # load demand and turning ratio
        demand_dict, turning_dict = self._load_demand_and_turning_ratio()

        # add additional class of segment and pipeline
        self._add_link_properties(demand_dict, turning_dict)

    def particle_filter(self):
        for link_id in self.links.keys():
            self.links[link_id].particle_forward(self.time_step)
            self.links[link_id].resample(self.particle_number)

            # left-turn spillover
            self.links[link_id].convert_coordinates(self.particle_number, self.grid_size)
        self.link_communication()

    def _close_simulator(self):
        print("here")
        if self._is_open:
            # close simulation
            traci.close()

            # output the figure
            if self.output_cost:
                self.output_network_performance()

            self._output_particle_time_space()

            # close simulator
            print("Close simulation with random seeds", self.sumo_seed, "done.")

            # reset the is open flag
            self._is_open = False
        delete_buffer_file(self.sumo_seed)

    def _output_particle_time_space(self):
        folder = env_config.output_trajs_folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        for link_id in self.links.keys():
            link = self.links[link_id]
            if link.link_type == "wrong" or link.link_type == "sink":
                continue

            pipelines = link.pipelines

            plt.figure(figsize=[19, 9])
            count = 0
            for pip_id in pipelines.keys():
                pipeline = pipelines[pip_id]
                # print(pipeline.lane_list)
                # print([self.lanes[val].allowable for val in pipeline.lane_list])
                plt.subplot(2, 2, count + 1)
                plt.title("Lane " + str(pipeline.id[-1]) + ", " + pipeline.direction +
                          ", " + str(np.round(pipeline.arrival_rate, 4)))
                plt.plot([0, self.time_step], [pipeline.start_dis, pipeline.start_dis], "k--")
                signal = pipeline.signal
                movement = pipeline.movement
                directions = pipeline.direction
                if directions is not None:
                    chosen_idx = 0
                    for i_d in range(len(directions)):
                        if directions[i_d] != "r":
                            chosen_idx = i_d
                            break
                    movement = self.signals[signal[chosen_idx]].movements[movement[chosen_idx]]
                    signal_state_list = movement.state_list
                    for i_s in range(len(signal_state_list)):
                        if signal_state_list[i_s]:
                            plt.plot([i_s, i_s + 1], [link.length + 7] * 2, "g")
                        else:
                            plt.plot([i_s, i_s + 1], [link.length + 7] * 2, "r")

                cv_trajs = pipeline.cv_trajs
                particles = pipeline.particle_history

                particle_t = []
                particle_s = []
                for itime in particles.keys():
                    locations = particles[itime]
                    for val in locations:
                        particle_t += [itime] * len(val)
                        particle_s += val

                for vid in cv_trajs.keys():
                    [vt, vd] = cv_trajs[vid]
                    plt.plot(vt, vd, "-", color="b", linewidth=1)

                for vid in pipeline.non_cv_trajs.keys():
                    [vt, vd] = pipeline.non_cv_trajs[vid]
                    plt.plot(vt, vd, "--", color="b", linewidth=1, alpha=1)
                plt.plot(particle_t, particle_s, ".", color="k", alpha=1.0 / self.particle_number,
                         markersize=2)
                plt.xlim([0, self.time_step])
                plt.ylim([0, link.length + 20])
                count += 1
            overall_title = link.link_id + "  " + link.link_type

            plt.suptitle(overall_title)
            plt.savefig(os.path.join(folder, overall_title + ".png"), dpi=300)
            # plt.show()
            plt.close()

            plt.figure(figsize=[19, 9])
            count = 0
            for pip_id in pipelines.keys():
                pipeline = pipelines[pip_id]
                plt.subplot(2, 2, count + 1)
                plt.imshow(np.transpose(pipeline.density_matrix),
                           cmap="binary", aspect="auto", vmin=0, vmax=np.ceil(self.grid_size / 7))
                count += 1
                plt.ylim([np.floor(link.length / self.grid_size), -0.5])
            overall_title = link.link_id + "  " + link.link_type + "   ground truth"

            plt.suptitle(overall_title)
            plt.savefig(os.path.join(folder, link.link_id + "_density.png"), dpi=300)
            # plt.show()
            plt.close()

            plt.figure(figsize=[19, 9])
            count = 0
            for pip_id in pipelines.keys():
                pipeline = pipelines[pip_id]
                plt.subplot(2, 2, count + 1)
                plt.imshow(np.transpose(pipeline.real_time_density),
                           cmap="binary", aspect="auto", vmin=0, vmax=np.ceil(self.grid_size / 7))
                count += 1
                plt.ylim([np.floor(link.length / self.grid_size), -0.5])
            overall_title = link.link_id + "  " + link.link_type + "   real-time"

            plt.suptitle(overall_title)
            plt.savefig(os.path.join(folder, link.link_id + "_real-time_density.png"), dpi=300)
            # plt.show()
            plt.close()

            plt.figure(figsize=[19, 9])
            count = 0
            for pip_id in pipelines.keys():
                pipeline = pipelines[pip_id]
                plt.subplot(2, 2, count + 1)
                plt.imshow(np.transpose(pipeline.observed_density),
                           cmap="binary", aspect="auto", vmin=0, vmax=np.ceil(self.grid_size / 7))
                count += 1
                plt.ylim([np.floor(link.length / self.grid_size), -0.5])
            overall_title = link.link_id + "  " + link.link_type + "   observed"

            plt.suptitle(overall_title)
            plt.savefig(os.path.join(folder, link.link_id + "_observed_density.png"), dpi=300)
            # plt.show()
            plt.close()
            print("Output", overall_title, "done")

    def link_communication(self):
        """
        update the downstream constraints

        :return:
        """
        # initiate and clean the upstream arrival
        for link_id in self.links.keys():
            link = self.links[link_id]
            if link.link_type == "source":
                continue
            pipelines = link.pipelines
            for pip_id in pipelines.keys():
                self.links[link_id].pipelines[pip_id].upstream_arrival = [0] * self.particle_number

        for link_id in self.links.keys():
            link = self.links[link_id]
            pipelines = link.pipelines
            for pip_id in pipelines.keys():
                pipeline = pipelines[pip_id]
                directions = pipeline.direction
                if directions is None:
                    continue
                signals = pipeline.signal
                movements = pipeline.movement
                downstream_links = pipeline.downstream_links
                downstream_pips = pipeline.downstream_pipelines

                outflow = pipeline.outflow
                outflow_flag = outflow is not None
                # print(outflow)
                downstream_dis = {}

                for i_dir in range(len(directions)):
                    downstream_pip_id = downstream_pips[i_dir]
                    direction = directions[i_dir]
                    tail_length = pipeline.tail_length[i_dir]
                    signal_state = self.signals[signals[i_dir]].movements[movements[i_dir]].state_list[-1]
                    downstream_pip = self.links[downstream_links[i_dir]].pipelines[downstream_pip_id]
                    downstream_particles = downstream_pip.particles

                    local_downstream_dis = [link.length + 7] * self.particle_number
                    for ip in range(self.particle_number):
                        if outflow_flag:
                            if outflow[direction][ip] is not None:
                                self.links[downstream_links[i_dir]].pipelines[downstream_pip_id].upstream_arrival[
                                    ip] = outflow[direction][ip] + 1

                        # if the signal is green
                        if signal_state:
                            local_downstream_dis[ip] = 2 * link.length
                            if len(downstream_particles["start"][ip][0]) == 0:
                                if len(downstream_pip.vehicles[0]) == 0:
                                    continue
                                else:
                                    dis = downstream_pip.vehicles[1][0]
                                    local_downstream_dis[ip] = dis + link.length + tail_length
                            else:
                                local_downstream_dis[ip] = \
                                    downstream_particles["start"][ip][0][0] + link.length + tail_length

                    downstream_dis[direction] = local_downstream_dis

                # dump the downstream distance to the current pipeline
                self.links[link_id].pipelines[pip_id].downstream_dis = downstream_dis

    def _add_link_properties(self, demand_dict, turning_dict):
        signalized_junction_list = list(self.signals.keys())
        for link_id in self.links.keys():
            link = self.links[link_id]

            link.__class__ = ParticleLink
            link.add_attributes()

            pipelines = link.pipelines
            self.links[link_id].ingress_lanes = len(self.edges[link.edge_list[0]].lanes_list)
            car_following_model = SimplifiedModel()
            car_following_model.free_flow_speed = link.speed
            self.links[link_id].car_following = car_following_model

            # convert the link pipeline to a class
            new_pipeline_dict = {}
            for pip_idx in pipelines.keys():
                lane_list = pipelines[pip_idx]
                pipeline_length = 0
                for lane_id in lane_list:
                    lane = self.lanes[lane_id]
                    if lane.allowable:
                        pipeline_length += lane.length
                pipeline = PipeLine(lane_list[-1], lane_list, self.particle_number)
                pipeline.length = pipeline_length
                pipeline.start_dis = link.length - pipeline_length
                if pipeline.start_dis < 5:
                    link.external_arrival_lanes += 1
                new_pipeline_dict[pipeline.id] = pipeline

            # there is one abnormal
            if len(link.segments) > 2:
                print(link_id, "is not correct for the left turn pipeline...")

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
                    self.links[link_id].particle_arrival = \
                        (1 - self.penetration_rate) * demand_dict[start_edge] * self.relative_demand

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

                    via_lane = downstream_info["via"]
                    internal_length = self.lanes[via_lane].length

                    downstream_edge = self.lanes[downstream_lane].edge_id
                    probability = turning_dict[end_edge][downstream_edge]

                    self.lanes[lane_id].downstream_lanes[downstream_lane]["prob"] = probability
                    pipeline_id = downstream_info["upstream_lane"]
                    if not (pipeline_id in new_pipeline_dict.keys()):
                        exit("There is something wrong loading the turning ratio to the pipeline...")

                    if downstream_info["dir"] == "L":
                        downstream_info["dir"] = "l"
                    if downstream_info["dir"] == "s":
                        self.links[link_id].through = probability
                        self.links[link_id].through_pipelines.append(pipeline_id)
                    elif downstream_info["dir"] == "l" or downstream_info["dir"] == "L":
                        self.links[link_id].left_turn = probability
                        self.links[link_id].left_pipelines.append(pipeline_id)
                    elif downstream_info["dir"] == "r":
                        self.links[link_id].right_turn = probability
                        self.links[link_id].right_pipelines.append(pipeline_id)
                    else:
                        exit("Direction " + str(downstream_info["dir"]) + " not recognized!!")

                    # dump the other information: direction, traffic light, link, etc...
                    if new_pipeline_dict[pipeline_id].direction is None:
                        new_pipeline_dict[pipeline_id].direction = ""
                        new_pipeline_dict[pipeline_id].signal = []
                        new_pipeline_dict[pipeline_id].movement = []
                        new_pipeline_dict[pipeline_id].tail_length = []

                    new_pipeline_dict[pipeline_id].direction += downstream_info["dir"]
                    new_pipeline_dict[pipeline_id].signal.append(downstream_info["tl"])
                    new_pipeline_dict[pipeline_id].movement.append(int(downstream_info["link_id"]))
                    new_pipeline_dict[pipeline_id].tail_length.append(internal_length)
                    new_pipeline_dict[pipeline_id].exit_length[downstream_info["dir"]] = \
                        link.length + internal_length

            # load the new pipeline object to the link
            self.links[link_id].pipelines = new_pipeline_dict

        # add the downstream pipeline
        for link_id in self.links.keys():
            link = self.links[link_id] 
            pipelines = link.pipelines
            for pip_id in pipelines.keys():
                pipeline = pipelines[pip_id]

                signal_list = pipeline.signal
                if signal_list is None:
                    continue
                movement_list = pipeline.movement

                downstream_num = len(movement_list)
                downstream_pips = []
                downstream_links = []
                for t in range(downstream_num):
                    local_movement = movement_list[t]
                    local_signal = signal_list[t]
                    movement = self.signals[local_signal].movements[local_movement]
                    downstream_lane = movement.exit_lane
                    downstream_link = movement.exit_link
                    downstream_pipeline = self.links[downstream_link].get_lane_belonged_pipeline(downstream_lane)
                    downstream_pips.append(downstream_pipeline)
                    downstream_links.append(downstream_link)
                self.links[link_id].pipelines[pip_id].downstream_pipelines = downstream_pips
                self.links[link_id].pipelines[pip_id].downstream_links = downstream_links

        # set the arrival rate for each pipeline
        for link_id in self.links.keys():
            link = self.links[link_id]
            pipelines = link.pipelines
            if link.link_type != "source":
                continue
            for pip_id in pipelines.keys():
                pipeline = pipelines[pip_id]
                if pipeline.start_dis < 5:
                    pipeline.arrival_rate = link.particle_arrival / link.external_arrival_lanes

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
                        vehicle.segment_list.append(segment_id)
                        segment_assigned = True
                        break
                if not segment_assigned:
                    original_segment = self.vehicles[vehicle_id].segment_list[-1]
                    vehicle.segment_list.append(original_segment)

                # get the pipeline id
                pipeline_assigned = False
                for pipeline_id in pipelines:
                    pipeline_lanes = pipelines[pipeline_id].lane_list
                    if vehicle_lane in pipeline_lanes:
                        vehicle.pipeline_list.append(pipeline_id)
                        pipeline_assigned = True

                        if not (pipeline_id in pipeline_vehicle_dict.keys()):
                            pipeline_vehicle_dict[pipeline_id] = {"vehicles": [], "dis": []}
                        pipeline_vehicle_dict[pipeline_id]["vehicles"].append(vehicle_id)
                        pipeline_vehicle_dict[pipeline_id]["dis"].append(vehicle_pos)

                if not pipeline_assigned:
                    original_pipeline = self.vehicles[vehicle_id].pipeline_list[-1]
                    vehicle.pipeline_list.append(original_pipeline)
                    if not (original_pipeline in pipeline_vehicle_dict.keys()):
                        pipeline_vehicle_dict[original_pipeline] = {"vehicles": [], "dis": []}
                    pipeline_vehicle_dict[original_pipeline]["vehicles"].append(vehicle_id)
                    pipeline_vehicle_dict[original_pipeline]["dis"].append(vehicle_pos)

                # detect the lane-changing event
                if vehicle.cv_type:
                    if len(vehicle.link_list) >= 2:
                        if len(vehicle.pipeline_list) >= 2:
                            if vehicle.link_list[-1] == vehicle.link_list[-2]:                # remain in the same link
                                if vehicle.pipeline_list[-1] != vehicle.pipeline_list[-2]:    # different pipeline
                                    self.links[link_id].lane_change_events[vehicle_id] = \
                                        [vehicle.pipeline_list[-2], vehicle.pipeline_list[-1],
                                         vehicle.link_pos_list[-1]]

            # sort the vehicle within different pipelines
            for pipeline_id in link.pipelines.keys():
                pipeline = link.pipelines[pipeline_id]

                # dump the current vehicle to the previous vehicle
                self.links[link_id].pipelines[pipeline_id].previous_vehicles = \
                    pipeline.vehicles
                if not (pipeline_id in pipeline_vehicle_dict.keys()):
                    self.links[link_id].pipelines[pipeline_id].vehicles = [[], []]
                    continue

                distance_list = pipeline_vehicle_dict[pipeline_id]["dis"]
                vehicle_id_list = pipeline_vehicle_dict[pipeline_id]["vehicles"]
                sequence = np.argsort(distance_list)
                new_vehicle_list = []
                new_distance_list = []
                for sdx in sequence:
                    new_vehicle_list.append(vehicle_id_list[sdx])
                    new_distance_list.append(distance_list[sdx])

                pipeline.complete_vehicles = [new_vehicle_list, new_distance_list]
                # dump the vehicle to the pipeline
                cv_list = []
                non_cv_list = []
                cv_distance_list = []
                non_cv_distance_list = []
                for vid in new_vehicle_list:
                    if self.vehicles[vid].cv_type:
                        cv_list.append(vid)
                        cv_distance_list.append(self.vehicles[vid].link_pos_list[-1])
                    else:
                        non_cv_list.append(vid)
                        non_cv_distance_list.append(self.vehicles[vid].link_pos_list[-1])
                pipeline.vehicles = [cv_list, cv_distance_list]

                # dump the vehicle to the pipeline dict
                for vdx in range(len(cv_list)):
                    vid = cv_list[vdx]
                    vdis = cv_distance_list[vdx]
                    if not (vid in pipeline.cv_trajs.keys()):
                        self.links[link_id].pipelines[pipeline_id].cv_trajs[vid] = [[], []]
                    pipeline.cv_trajs[vid][0].append(self.time_step)
                    pipeline.cv_trajs[vid][1].append(vdis)

                for vdx in range(len(non_cv_list)):
                    vid = non_cv_list[vdx]
                    vdis = non_cv_distance_list[vdx]
                    if not (vid in pipeline.non_cv_trajs.keys()):
                        self.links[link_id].pipelines[pipeline_id].non_cv_trajs[vid] = [[], []]
                    pipeline.non_cv_trajs[vid][0].append(self.time_step)
                    pipeline.non_cv_trajs[vid][1].append(vdis)

                new_vehicle_list = [None] + new_vehicle_list + [None]
                new_distance_list = [0] + new_distance_list + [link.length]

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

        for link_id in self.links.keys():
            self.links[link_id].update_turning_dict()

    def _get_coordinates(self, link_id, pip_id, link_pos_list):
        link = self.links[link_id]
        pipeline = link.pipelines[pip_id]
        lane_list = pipeline.lane_list

        x_list = []
        y_list = []
        for dis in link_pos_list:
            pip_dis = dis - pipeline.start_dis
            for lane_id in lane_list:
                lane = self.lanes[lane_id]
                if pip_dis <= lane.length:
                    x, y = self._get_specific_location(lane.shape[0], lane.shape[1], pip_dis)
                    if x is None:
                        continue
                    x_list.append(x)
                    y_list.append(y)
                    continue
                pip_dis -= lane.length
        return [x_list, y_list]

    @staticmethod
    def _get_specific_location(x_list, y_list, length):
        x_diff = np.abs(np.diff(x_list))
        y_diff = np.abs(np.diff(y_list))
        distance_list = np.sqrt(np.square(x_diff) + np.square(y_diff))
        for idx in range(len(distance_list)):
            local_length = distance_list[idx]
            if local_length >= length:
                proportion = length / local_length
                x = x_list[idx] * (1 - proportion) + x_list[idx + 1] * proportion
                y = y_list[idx] * (1 - proportion) + y_list[idx + 1] * proportion
                return x, y
            else:
                length -= local_length
        return None, None


class ParticleLink(Link):
    def __init__(self):
        super().__init__()

        self.link_type = None
        self.lane_change_events = None

        self.external_arrival_lanes = 0
        self.ingress_lanes = None
        self.demand = None
        self.particle_arrival = None

        # turning ratio
        self.through = 0
        self.left_turn = 0
        self.right_turn = 0

        self.left_pipelines = []
        self.right_pipelines = []
        self.through_pipelines = []

        self.turning_info = None
        self.car_following = None

    def add_attributes(self):
        # turning ratio
        self.through = 0
        self.left_turn = 0
        self.right_turn = 0

        self.external_arrival_lanes = 0
        self.left_pipelines = []
        self.right_pipelines = []
        self.through_pipelines = []
        self.turning_info = None
        self.particle_arrival = None

    def particle_forward(self, time_step):
        if self.link_type == "sink" or self.link_type == "wrong":
            return
        pipelines = self.pipelines
        link_cv_list = []

        # ge the full list
        for pip_id in pipelines.keys():
            pipeline = pipelines[pip_id]
            [pr_cv_list, _] = pipeline.previous_vehicles
            link_cv_list += pr_cv_list

        # remove exit vehicles
        for pip_id in pipelines.keys():
            pipeline = pipelines[pip_id]

            # remove exit vehicles
            particle_key_list = list(pipeline.particles.keys())
            if len(particle_key_list) > 1:
                if not (particle_key_list[-1] in link_cv_list):
                    # print("remove", particle_key_list[-1], "from", pip_id, "at")
                    del self.pipelines[pip_id].particles[particle_key_list[-1]]

        # perform a one-slot car following (prediction)
        self.step()

        for pip_id in pipelines.keys():
            pipeline = pipelines[pip_id]

            # [pr_cv_list, pr_cv_distance_list] = pipeline.previous_vehicles
            [cv_list, cv_distance_list] = pipeline.vehicles

            new_arrival = False
            # event: new arrival
            if len(cv_list) > 0:
                if not (cv_list[0] in link_cv_list):
                    new_arrival = True

            # if there is no new arrival cv, then generate the arrival for particle
            if not new_arrival:
                self.pipelines[pip_id].generate_new_arrival(self.turning_info)
            else:
                # update the pipeline particle accordingly
                self.pipelines[pip_id].new_cv_arrived(cv_list[0], cv_distance_list[0])

        # deal with the lane changing
        self.sort_lane_changing_events()
        if len(self.lane_change_events) > 0:
            # print(self.lane_change_events)
            for cv_id in self.lane_change_events.keys():
                [from_pip, to_pip, cv_dis] = self.lane_change_events[cv_id]
                self.pipelines[from_pip].remove_cv(cv_id)
                self.pipelines[to_pip].insert_cv(cv_id, cv_dis)

        # save the particle to memory
        for pip_id in self.pipelines.keys():
            self.pipelines[pip_id].save_particle(time_step)

    def resample(self, particle_number):
        if self.link_type == "wrong" or self.link_type == "sink":
            return
        for pip_id in self.pipelines.keys():
            pipeline = self.pipelines[pip_id]
            [cv_list, cv_dis] = pipeline.vehicles
            [pr_cv_list, pr_cv_dis] = pipeline.previous_vehicles

            cv_dis_dict = {}
            for iv in range(len(cv_list)):
                cv_dis_dict[cv_list[iv]] = cv_dis[iv]
            pr_dis_dict = {}
            for iv in range(len(pr_cv_list)):
                pr_dis_dict[pr_cv_list[iv]] = pr_cv_dis[iv]

            # determine the weight of the particles
            particle_weights = [1 for temp in range(particle_number)]
            exit_length = np.min([pipeline.exit_length[val] for val in pipeline.exit_length.keys()] + [10000])

            particles = pipeline.particles
            vehicle_list = list(particles.keys())[1:] + ["end"]
            pr_dis_dict["end"] = exit_length

            if len(vehicle_list) <= 2:
                continue
            for vid in range(len(vehicle_list) - 1):
                local_vid = vehicle_list[vid]
                if not (local_vid in cv_list):
                    continue
                next_vid = vehicle_list[vid + 1]
                particle = particles[local_vid]

                cv_headway = pr_dis_dict[next_vid] - pr_dis_dict[local_vid]
                following_speed = cv_dis_dict[local_vid] - pr_dis_dict[local_vid]

                for pdx in range(particle_number):
                    [locations, _] = particle[pdx]
                    if len(locations) > 1:
                        headway = locations[0] - pr_dis_dict[local_vid]
                    else:
                        headway = cv_headway
                    local_weight = self.car_following.get_weight(headway, following_speed)
                    particle_weights[pdx] *= local_weight

            resampled_sequence = self.get_resample_particle(particle_weights)

            new_particles = {}
            for vid in particles.keys():
                if not (vid in new_particles.keys()):
                    new_particles[vid] = []
                for pdx in range(particle_number):
                    new_particles[vid].append(particles[vid][resampled_sequence[pdx]])

            # backward update for the particle history
            pipeline.particle_history_backward_update(resampled_sequence)
            pipeline.particles = new_particles

    @staticmethod
    def get_resample_particle(particle_weights):
        # normalize the weight
        total_weights = np.sum(particle_weights)
        # print("total weights", total_weights)
        if total_weights < 1e-6:
            return range(len(particle_weights))

        weight_list = [val / total_weights for val in particle_weights]
        cdf_curve = np.cumsum(weight_list)

        # re-sample
        random_number = uniform.rvs()
        new_particles = []

        sample_size = len(particle_weights)

        for idx in range(sample_size):
            current_val = (random_number + idx) / sample_size
            for jdx in range(len(cdf_curve)):
                current_weight = cdf_curve[jdx]
                if current_weight >= current_val:
                    new_particles.append(jdx)
                    break
        return new_particles

    def convert_coordinates(self, particle_number, grid_size):
        for pip_id in self.pipelines.keys():
            pipeline = self.pipelines[pip_id]
            [_, cv_distance] = pipeline.vehicles
            vehicles_locations = [deepcopy(cv_distance) for val in range(particle_number)]
            particles = pipeline.particles
            for cv_id in particles.keys():
                cv_particles = particles[cv_id]
                for ip in range(particle_number):
                    [locs, _] = cv_particles[ip]
                    vehicles_locations[ip] += locs

            # spill over detection
            vehicles_nums = [len(val) for val in vehicles_locations]
            maximum_nums = pipeline.length / 7 + 2
            spillover = [val > maximum_nums for val in vehicles_nums]
            spillover_prob = np.average(spillover)
            if spillover_prob > 0.5:
                # print("spillover alert", pipeline.length, self.link_id, vehicles_nums)
                pass
            pipeline.spillover = spillover
            pipeline.spillover_prob.append(spillover_prob)

            # convert the Eulerian coordinates
            [_, all_distance] = pipeline.complete_vehicles

            time_key = list(pipeline.particle_history.keys())
            if len(time_key) == 0:
                particle_locations = []
            else:
                temp_par_locs = pipeline.particle_history[time_key[-1]]
                particle_locations = []
                for val in temp_par_locs:
                    particle_locations += val
            link_length = self.length

            cell_numbers = int(np.ceil(pipeline.length / grid_size))
            real_time_density_vector = [0 for val in range(cell_numbers)]

            for par_loc in particle_locations:
                if par_loc >= link_length:
                    # real_time_density_vector[0] += 1 / particle_number
                    pass
                elif par_loc <= 0:
                    real_time_density_vector[-1] += 1 / particle_number
                else:
                    add_index = int((link_length - par_loc) / grid_size)
                    if add_index >= cell_numbers:
                        real_time_density_vector[-1] += 1 / particle_number
                    else:
                        real_time_density_vector[add_index] += 1 / particle_number

            observed_density = [0 for val in range(cell_numbers)]
            for val in cv_distance:
                if val >= link_length:
                    # real_time_density_vector[0] += 1
                    # observed_density[0] += 1
                    pass
                elif val <= 0:
                    real_time_density_vector[-1] += 1
                    observed_density[-1] += 1
                else:
                    add_index = int((link_length - val) / grid_size)
                    if add_index >= cell_numbers:
                        real_time_density_vector[-1] += 1
                        observed_density[-1] += 1
                    else:
                        real_time_density_vector[add_index] += 1
                        observed_density[add_index] += 1
            pipeline.real_time_density.append(real_time_density_vector)
            pipeline.observed_density.append(observed_density)

            density_vector = [0 for val in range(cell_numbers)]
            for val in all_distance:
                if val >= link_length:
                    # density_vector[0] += 1
                    pass
                elif val <= 0:
                    density_vector[-1] += 1
                else:
                    add_index = int((link_length - val) / grid_size)
                    if add_index >= cell_numbers:
                        density_vector[-1] += 1
                    else:
                        density_vector[add_index] += 1
            pipeline.density_matrix.append(density_vector)

    def sort_lane_changing_events(self):
        if len(self.lane_change_events) <= 1:
            return
        new_lane_changing_events = {}
        v_list = []
        dis_list = []
        for v_id in self.lane_change_events.keys():
            v_list.append(v_id)
            dis_list.append(self.lane_change_events[v_id][2])
        sequence_list = np.argsort(dis_list)
        new_v_list = []
        for s in sequence_list:
            new_v_list.append(v_list[s])
        for v_id in new_v_list:
            new_lane_changing_events[v_id] = self.lane_change_events[v_id]
        self.lane_change_events = new_lane_changing_events

    def update_turning_dict(self):
        if self.link_type == "sink":
            self.turning_info = None
            return

        ratio_list = [self.left_turn, self.through, self.right_turn]
        pipeline_list = [self.left_pipelines, self.through_pipelines, self.right_pipelines]
        self.turning_info = [ratio_list, pipeline_list]

    def step(self):
        pipeline_dict = {}
        for pip_id in self.pipelines.keys():
            pipeline = self.pipelines[pip_id]
            pipeline_dict[pipeline.index] = pipeline.id

        # stochastic car-following model
        for pip_id in self.pipelines.keys():
            pipeline = self.pipelines[pip_id]
            particles = pipeline.particles
            direction = pipeline.direction
            pipeline.outflow = {}

            for i_dir in direction:
                self.pipelines[pip_id].outflow[i_dir] = [None] * pipeline.particle_number

            [cv_list, cv_distance_list] = deepcopy(pipeline.previous_vehicles)
            downstream_num = len(direction)
            downstream_dis = pipeline.downstream_dis

            particle_keys = list(particles.keys())

            # check cv list
            if particle_keys[1:] != cv_list:
                if set(particle_keys[1:]) == set(cv_list):
                    new_keys_list = ["start"] + cv_list
                    new_particles = {}
                    for i_v in new_keys_list:
                        new_particles[i_v] = particles[i_v]
                    self.pipelines[pip_id].particles = new_particles
                    particle_keys = new_keys_list
                else:
                    print("\n report error information")
                    print(self.link_id, pip_id, "not consistent")
                    print(cv_list, cv_distance_list)
                    print(particle_keys[1:])
                    print(self.lane_change_events)
                    exit("not consistent error!")

            for i_p in range(len(cv_list) + 1):
                if i_p == len(cv_list):
                    mid_flag = False
                    last_distance = self.length
                else:
                    mid_flag = True
                    last_distance = cv_distance_list[i_p]
                cv_id = particle_keys[i_p]
                local_particles = particles[cv_id]

                for pdx in range(pipeline.particle_number):
                    [location_list, lane_change_list] = local_particles[pdx]

                    # skip when there is no vehicle of the particle
                    if len(location_list) == 0:
                        continue

                    new_lane_change_list = lane_change_list

                    # if not the last, only car-following
                    if mid_flag:
                        new_location_list = \
                            self.car_following.sample_next_locations(location_list + [last_distance], None, 1)
                    else:
                        # determine the turning flag
                        direction_flag = direction
                        if downstream_num > 1:
                            if lane_change_list[-1] is None:
                                direction_flag = "s"
                            else:
                                direction_flag = "r"

                        if downstream_dis is None:
                            last_distance = 2 * self.length
                        else:
                            last_distance = downstream_dis[direction_flag][pdx]

                        new_location_list = \
                            self.car_following.sample_next_locations(location_list + [last_distance], None, 1)

                        exit_length = pipeline.exit_length[direction_flag]
                        outreach_length = new_location_list[-1] - exit_length

                        # put the vehicle to out flow
                        if outreach_length > 0:
                            self.pipelines[pip_id].outflow[direction_flag][pdx] = outreach_length
                            new_location_list = new_location_list[:-1]
                            new_lane_change_list = lane_change_list[:-1]
                            location_list = location_list[:-1]
                            lane_change_list = lane_change_list[:-1]

                    # self.pipelines[pip_id].particles[cv_id][pdx] = [new_location_list, new_lane_change_list]
                    # continue

                    latest_locations = []
                    latest_lane_infos = []

                    # deal with the lane change from the inverse direction
                    spillover = False
                    for i_v in range(len(new_location_list)):
                        if spillover:                         # if spill over occurs, there will be no lane changing
                            location = location_list[::-1][i_v]
                            lane_change = lane_change_list[::-1][i_v]
                        else:
                            location = new_location_list[::-1][i_v]
                            lane_change = new_lane_change_list[::-1][i_v]

                        # keep in the lane
                        if lane_change is None:
                            latest_locations.append(location)
                            latest_lane_infos.append(lane_change)
                            continue
                        if lane_change == pipeline.index:
                            latest_locations.append(location)
                            latest_lane_infos.append(lane_change)
                            continue
                        current_index = pipeline.index
                        dest_pip_id = int(current_index + np.sign(lane_change - current_index))
                        dest_pip_id = pipeline_dict[dest_pip_id]
                        destination_pipeline = self.pipelines[dest_pip_id]
                        start_dis = destination_pipeline.start_dis

                        # keep in the lane
                        if location < start_dis - 7:
                            latest_locations.append(location)
                            latest_lane_infos.append(lane_change)
                            continue

                        [destination_cv, destination_dis] = destination_pipeline.vehicles
                        if spillover:
                            latest_locations.append(location_list[::-1][i_v])
                            latest_lane_infos.append(lane_change_list[::-1][i_v])
                            continue
                        else:
                            downstream_spillover = destination_pipeline.spillover[pdx]
                            if downstream_spillover:
                                spillover = True
                                latest_locations.append(location_list[::-1][i_v])
                                latest_lane_infos.append(lane_change_list[::-1][i_v])
                                continue
                            else:
                                # directly move the vehicle to the destination link
                                insert_index = 0
                                for i_dis in range(len(destination_dis)):
                                    if location > destination_dis[i_dis]:
                                        insert_index = i_dis
                                        break
                                destination_cv_list = ["start"] + destination_cv
                                insert_cv = destination_cv_list[insert_index]
                                [dest_dis, dest_lane] = self.pipelines[dest_pip_id].particles[insert_cv][pdx]

                                if len(dest_dis) == 0:
                                    new_dest_dis, new_dest_lane = [location], [lane_change]
                                else:
                                    insert_index = 0
                                    for i_dis in range(len(dest_dis)):
                                        if location < dest_dis[i_dis]:
                                            insert_index = i_dis
                                            break
                                    new_dest_dis = dest_dis[:insert_index] + [location] + dest_dis[insert_index:]
                                    new_dest_lane = dest_lane[:insert_index] + [lane_change] + dest_lane[
                                                                                               insert_index:]

                                self.pipelines[dest_pip_id].particles[insert_cv][pdx] = [new_dest_dis,
                                                                                         new_dest_lane]
                    self.pipelines[pip_id].particles[cv_id][pdx] = [latest_locations[::-1], latest_lane_infos[::-1]]

    def get_lane_belonged_pipeline(self, lane_id):
        for pip_id in self.pipelines.keys():
            pipeline = self.pipelines[pip_id]
            lane_list = pipeline.lane_list
            if lane_id in lane_list:
                return pip_id
        return None


class PipeLine(object):
    """
    pipeline is for each continuous lane
    """
    def __init__(self, pipeline_id, lane_list, particle_number):
        self.particle_number = particle_number
        self.id = pipeline_id
        self.index = int(pipeline_id[-1])
        self.lane_list = lane_list

        self.density_matrix = []
        self.real_time_density = []
        self.offline_density = []
        self.observed_density = []

        self.arrival_rate = 0
        self.length = None
        self.start_dis = None
        self.tail_length = None

        self.exit_length = {}
        self.direction = None
        self.signal = None
        self.movement = None
        self.downstream_links = None
        self.downstream_pipelines = None

        self.downstream_dis = None
        self.outflow = None
        self.upstream_arrival = None
        self.spillover = False
        self.spillover_prob = []

        # real-time state
        self.vehicles = [[], []]
        self.complete_vehicles = [[], []]
        self.previous_vehicles = [[], []]

        self.cv_trajs = {}
        self.non_cv_trajs = {}
        self.particle_history = {}

        # particles
        # start ----- cv1 ----- cv2 ----- ...
        # {"start": [[distance1 < distance2 <...], [dest_pip1, dest_pip2, ...]]}
        self.particles = {"start": [[[], []]] * particle_number}

    def new_cv_arrived(self, cv_id, cv_dis):
        old_keys = list(self.particles.keys())
        new_keys = old_keys[:1] + [cv_id] + old_keys[1:]
        self.previous_vehicles[0] = [cv_id] + self.previous_vehicles[0]
        self.previous_vehicles[1] = [cv_dis] + self.previous_vehicles[1]
        # print("add", cv_dis, cv_id, self.id)

        new_particles = {}
        for ik in new_keys:
            if ik == "start":
                new_particles[ik] = [[[], []]] * self.particle_number
                continue
            if ik == cv_id:
                new_particles[cv_id] = self.particles["start"]
                continue
            new_particles[ik] = self.particles[ik]
        self.particles = new_particles

    def generate_new_arrival(self, turning_info):
        [ratio_list, pip_list] = turning_info

        arrival_rate = self.arrival_rate
        # generate the arrival series
        if arrival_rate is not None and arrival_rate > 0:
            arrival_seeds = uniform.rvs(size=self.particle_number)
            arrival_list = [val < arrival_rate for val in arrival_seeds]
        else:
            if self.upstream_arrival is None:
                return
            arrival_list = self.upstream_arrival

        # generate the turning series
        turning_seeds = uniform.rvs(size=self.particle_number)
        pip_seeds = [int(val * 10) for val in turning_seeds]

        for ip in range(self.particle_number):
            if arrival_list[ip]:
                self.particles["start"][ip][0] = [arrival_list[ip] - 1] + self.particles["start"][ip][0]

                if turning_seeds[ip] < ratio_list[0]:
                    local_pips = [int(val[-1]) for val in pip_list[0]]
                    chosen_pip = local_pips[pip_seeds[ip] % len(local_pips)]
                    direction = "l"
                elif ratio_list[0] < turning_seeds[ip] < (ratio_list[0] + ratio_list[1]):
                    local_pips = [int(val[-1]) for val in pip_list[1]]
                    chosen_pip = local_pips[pip_seeds[ip] % len(local_pips)]
                    direction = "s"
                else:
                    local_pips = [int(val[-1]) for val in pip_list[2]]
                    chosen_pip = local_pips[pip_seeds[ip] % len(local_pips)]
                    direction = "r"

                # the vehicle will only turn when necessary
                if not (direction in self.direction):
                    self.particles["start"][ip][1] = [chosen_pip] + self.particles["start"][ip][1]
                else:
                    self.particles["start"][ip][1] = [None] + self.particles["start"][ip][1]
        return 0

    def insert_cv(self, cv_id, cv_dis):
        pr_distance_list = [0] + self.previous_vehicles[1]
        pr_vehicle_list = list(self.particles.keys())
        insert_cv_index = len(pr_distance_list)
        for idx in range(len(pr_distance_list)):
            if cv_dis < pr_distance_list[idx]:
                insert_cv_index = idx
                break
        new_cv_list = pr_vehicle_list[:insert_cv_index] + [cv_id] + pr_vehicle_list[insert_cv_index:]
        new_dis_list = pr_distance_list[:insert_cv_index] + [cv_dis] + pr_distance_list[insert_cv_index:]
        self.previous_vehicles[0] = new_cv_list[1:]
        self.previous_vehicles[1] = new_dis_list[1:]

        split_cv = new_cv_list[insert_cv_index - 1]
        if not (split_cv in self.particles.keys()):
            print("insert cv error", self.id)
            print(pr_distance_list, pr_vehicle_list)
            print(new_cv_list, new_dis_list, split_cv)
        split_particle = self.particles[split_cv]
        lead_particle = []
        lag_particle = []
        for ip in range(len(split_particle)):
            [dis_list, lane_list] = split_particle[ip]
            s_index = len(dis_list)
            for iv in range(len(dis_list)):
                if dis_list[iv] > cv_dis:
                    s_index = iv
            p_dis = dis_list[:s_index]
            p_lane = lane_list[:s_index]
            lag_dis = dis_list[s_index:]
            lag_lane = lane_list[s_index:]
            lead_particle.append([p_dis, p_lane])
            lag_particle.append([lag_dis, lag_lane])

        new_particle = {}
        for v_id in new_cv_list:
            if v_id == split_cv:
                new_particle[v_id] = lead_particle
                continue
            if v_id == cv_id:
                new_particle[v_id] = lag_particle
                continue
            # not inserted yet
            if not (v_id in self.particles.keys()):
                continue
            new_particle[v_id] = self.particles[v_id]

        self.particles = new_particle

    def remove_cv(self, cv_id):
        if not (cv_id in self.particles.keys()):
            exit("CV id " + cv_id + " not in the particle keys! (" + str(self.id) + ")")

        merge_cv_index = None
        cv_list = list(self.particles.keys())
        for idx in range(len(cv_list)):
            if cv_id == cv_list[idx]:
                merge_cv_index = idx
                break
        new_particle = self.merge(self.particles[cv_list[merge_cv_index - 1]],
                                  self.particles[cv_list[merge_cv_index]])

        self.particles[cv_list[merge_cv_index - 1]] = new_particle
        del self.particles[cv_id]
        del self.previous_vehicles[0][merge_cv_index - 1]
        del self.previous_vehicles[1][merge_cv_index - 1]

    def save_particle(self, time_step):
        particles = self.particles
        location_list = [[] for fvd in range(self.particle_number)]
        for fvd in particles.keys():
            for ip in range(self.particle_number):
                current_locs = particles[fvd][ip]
                location_list[ip] += current_locs[0]
        self.particle_history[time_step] = location_list

    def particle_history_backward_update(self, new_sequence, backward_duration=-1):
        history_times = list(self.particle_history.keys())
        if backward_duration > 0:
            correct_times = history_times[max(len(history_times) - backward_duration, 0):]
        else:
            correct_times = history_times
        update_dict = {}
        for co_t in correct_times:
            original_locations = self.particle_history[co_t]
            new_locations = []
            for ip in range(self.particle_number):
                new_locations.append(original_locations[new_sequence[ip]])
            update_dict[co_t] = new_locations

        # update the new history
        for i_t in update_dict.keys():
            self.particle_history[i_t] = update_dict[i_t]

    @staticmethod
    def merge(particle1, particle2):
        new_particle = []
        for ip in range(len(particle1)):
            new_particle.append([particle1[ip][0] + particle2[ip][0], particle1[ip][1] + particle2[ip][1]])
        return new_particle

