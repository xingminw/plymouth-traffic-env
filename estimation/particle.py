"""
inherit the whole class of traffic env and add a particle filter estimation part
"""

from abc import ABC
from time import sleep
from copy import deepcopy
from scipy.stats import uniform, norm
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
        self.particle_number = 1

        # number of grid size
        self.grid_size = 15

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
            self.links[link_id].resample(self.particle_number)
            self.links[link_id].particle_forward(self.time_step)

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

            # output single link time-space diagram and density matrix
            self._output_particle_time_space()

            # output the corridor time-space diagram and density matrix
            self._output_corridor_estimation()

            # close simulator
            print("Close simulation with random seeds", self.sumo_seed, "done.")

            # reset the is open flag
            self._is_open = False
        delete_buffer_file(self.sumo_seed)

    def _output_corridor_estimation(self, image_type="png"):
        folder = env_config.output_trajs_folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        w2e_list = env_config.corridor_w2e
        e2w_list = env_config.corridor_e2w

        intersection_name = env_config.intersection_name_list
        link_list = [w2e_list, e2w_list]
        intersection_name_list = [intersection_name, intersection_name[::-1]]

        for idx in range(2):
            local_links = link_list[idx]
            intersection_names = intersection_name_list[idx]

            # output the time-space diagram
            plt.figure(figsize=[10, 10], dpi=500)
            start_dis = 0
            buffer = 5
            landmark_list = []
            for link_id in local_links:
                link = self.links[link_id]
                pipelines = link.pipelines

                signal_state_list = None

                # plot the signal state
                enter_edge = link.edge_list[-1]
                edge = self.edges[enter_edge]
                lane_list = edge.lanes_list
                for lane_id in lane_list:
                    lane = self.lanes[lane_id]
                    if lane.controlled_movement is not None:
                        [signal_id, movement_id] = lane.controlled_movement
                        movement = self.signals[signal_id].movements[movement_id]

                        direction = movement.direction
                        if direction == "s":
                            signal_state_list = movement.state_list

                if signal_state_list is not None:
                    for jdx in range(len(signal_state_list)):
                        signal_state = signal_state_list[jdx]
                        x_list = [jdx, jdx + 1]
                        y_list = [start_dis + link.length] * 2
                        if signal_state >= 1:
                            plt.plot(x_list, y_list, "g", lw=1.8)
                        else:
                            plt.plot(x_list, y_list, "r", lw=1.8)

                for pip_id in pipelines.keys():
                    pipeline = pipelines[pip_id]
                    directions = pipeline.direction
                    if directions == "l":
                        continue

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
                        plt.plot(vt, [val + start_dis for val in vd], "-", color="b", linewidth=0.5)

                    for vid in pipeline.non_cv_trajs.keys():
                        [vt, vd] = pipeline.non_cv_trajs[vid]
                        plt.plot(vt, [val + start_dis for val in vd], "--", color="b", linewidth=0.5, alpha=1)

                    particle_s = [val + start_dis for val in particle_s]
                    plt.plot(particle_t, particle_s, ".", color="k", alpha=0.6 / self.particle_number,
                             markersize=0.5)
                start_dis = start_dis + link.length + buffer
                landmark_list.append(start_dis)
            landmark_list = landmark_list[:-1]
            plt.yticks(landmark_list, intersection_names)
            plt.xlim([0, self.time_step])
            plt.ylim([0, landmark_list[-1] + buffer * 5])
            # plt.tight_layout()
            plt.xlabel("Time (s)")
            plt.ylabel("Distance")
            plt.savefig(folder + "/corridor_ts_" + str(idx) + "." + image_type)
            plt.close()

            # output the ground density matrix
            start_num = 0
            landmark_list = []

            corridor_density_matrix = None
            corridor_observed_matrix = None
            corridor_estimation_matrix = None

            for link_id in local_links:
                link = self.links[link_id]
                pipelines = link.pipelines

                pip_counts = 0
                agg_density_matrix = None
                agg_observed_matrix = None
                agg_estimation_matrix = None
                for pip_id in pipelines.keys():
                    pipeline = pipelines[pip_id]
                    if pipeline.start_dis > 5:
                        continue
                    pip_counts += 1
                    density_matrix = np.transpose(pipeline.density_matrix)
                    observed_matrix = np.transpose(pipeline.observed_density)
                    estimation_matrix = np.transpose(pipeline.real_time_density)

                    if agg_density_matrix is None:
                        agg_density_matrix = density_matrix
                        agg_observed_matrix = observed_matrix
                        agg_estimation_matrix = estimation_matrix
                    else:
                        agg_density_matrix = density_matrix + agg_density_matrix
                        agg_observed_matrix = observed_matrix + agg_observed_matrix
                        agg_estimation_matrix = estimation_matrix + agg_estimation_matrix
                agg_density_matrix = agg_density_matrix / pip_counts
                agg_observed_matrix = agg_observed_matrix / pip_counts
                agg_estimation_matrix = agg_estimation_matrix / pip_counts

                if corridor_density_matrix is None:
                    corridor_density_matrix = agg_density_matrix
                    corridor_observed_matrix = agg_observed_matrix
                    corridor_estimation_matrix = agg_estimation_matrix
                else:
                    corridor_estimation_matrix = \
                        np.concatenate((agg_estimation_matrix, corridor_estimation_matrix), axis=0)
                    corridor_density_matrix = \
                        np.concatenate((agg_density_matrix, corridor_density_matrix), axis=0)
                    corridor_observed_matrix = \
                        np.concatenate((agg_observed_matrix, corridor_observed_matrix), axis=0)

                (m, n) = np.shape(agg_density_matrix)
                start_num += m
                landmark_list.append(start_num)

            plt.figure(figsize=[10, 10], dpi=300)
            plt.yticks(landmark_list, intersection_names)
            plt.ylim([0, landmark_list[-2] + 5])
            plt.imshow(corridor_estimation_matrix[::-1],
                       cmap="binary", aspect="auto", vmin=0, vmax=np.ceil(self.grid_size / 7))
            plt.savefig(folder + "/corridor_estimation_" + str(idx) + "." + image_type)
            plt.close()

            plt.figure(figsize=[10, 10], dpi=300)
            plt.yticks(landmark_list, intersection_names)
            plt.ylim([0, landmark_list[-2] + 5])
            plt.imshow(corridor_density_matrix[::-1],
                       cmap="binary", aspect="auto", vmin=0, vmax=np.ceil(self.grid_size / 7))
            plt.savefig(folder + "/corridor_density_" + str(idx) + "." + image_type)
            plt.close()

            plt.figure(figsize=[10, 10], dpi=300)
            plt.yticks(landmark_list, intersection_names)
            plt.ylim([0, landmark_list[-2] + 5])
            plt.imshow(corridor_observed_matrix[::-1],
                       cmap="binary", aspect="auto", vmin=0, vmax=np.ceil(self.grid_size / 7))
            plt.savefig(folder + "/corridor_observed_" + str(idx) + "." + image_type)
            plt.close()

    def _output_particle_time_space(self):
        folder = env_config.output_trajs_folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        # draw the time-space diagram
        for link_id in self.links.keys():
            link = self.links[link_id]
            if link.link_type == "wrong" or link.link_type == "sink":
                continue

            pipelines = link.pipelines

            plt.figure(figsize=[19, 9])
            count = 0
            for pip_id in pipelines.keys():
                pipeline = pipelines[pip_id]
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
            print("Output", link.link_id, "done")

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
                self.links[link_id].pipelines[pip_id].upstream_arrival = [None for val in range(self.particle_number)]

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

                    local_downstream_dis = [link.length + 7 for val in range(self.particle_number)]
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

        # set the arrival rate for each pipeline and also update the turning dict
        for link_id in self.links.keys():
            link = self.links[link_id]
            link.update_turning_dict()
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

            # initiate the real-time storage info for pipeline class
            for pip_id in pipelines.keys():
                pipeline = pipelines[pip_id]

                # dump the current vehicle to the previous vehicle
                pipeline.previous_vehicles = pipeline.vehicles

                # to be updated with new observation coming in
                pipeline.complete_vehicles = [[], []]
                pipeline.vehicles = [[], []]

            if link_id in link_vehicle_dict.keys():
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
                        if len(vehicle.link_list) >= 2 and len(vehicle.pipeline_list) >= 2:
                            if vehicle.link_list[-1] == vehicle.link_list[-2] and vehicle.pipeline_list[-1] != \
                                    vehicle.pipeline_list[-2]:        # remain in the same link but different pipeline
                                # dump the lane change info to the link
                                link.lane_change_events[vehicle_id] = \
                                    [vehicle.pipeline_list[-2], vehicle.pipeline_list[-1],
                                     vehicle.link_pos_list[-1]]

                                # # dump the lane change info also to the pipeline
                                # link.pipelines[vehicle.pipeline_list[-2]].lane_change_out.append(vehicle_id)
                                # link.pipelines[vehicle.pipeline_list[-1]].lane_change_in.append(vehicle_id)

                # sort the vehicle within different pipelines
                for pipeline_id in link.pipelines.keys():
                    pipeline = link.pipelines[pipeline_id]

                    if pipeline_id in pipeline_vehicle_dict.keys():
                        # sort all the vehicles within the pipeline and dump it to the complete_vehicles
                        distance_list = pipeline_vehicle_dict[pipeline_id]["dis"]
                        vehicle_id_list = pipeline_vehicle_dict[pipeline_id]["vehicles"]
                        sequence = np.argsort(distance_list)
                        new_vehicle_list = []
                        new_distance_list = []
                        for sdx in sequence:
                            new_vehicle_list.append(vehicle_id_list[sdx])
                            new_distance_list.append(distance_list[sdx])
                        pipeline.complete_vehicles = [new_vehicle_list, new_distance_list]

                        # dump the vehicle to the pipeline vehicle (cv)
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

                        # dump the vehicle trajectory to the pipeline, cv and non-cv separately
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
        if self.link_type == "wrong" or self.link_type == "sink":
            return
        pipelines = self.pipelines

        # perform a one-slot car following (prediction)
        self.step()

        for pip_id in pipelines.keys():
            pipeline = pipelines[pip_id]
            # generate the new arrival
            pipeline.generate_new_arrival(self.turning_info)

            # update the particle dict
            pipeline.update_particle_dict()

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

            critical_cv = {}
            for vdx in range(len(pr_cv_list)):
                pr_vid = pr_cv_list[vdx]
                if not (pr_vid in cv_list):
                    continue

                if vdx == len(pr_cv_list) - 1:
                    downstream_dis = pipeline.downstream_dis
                    if downstream_dis is None:
                        downstream_dis = [2 * self.length]
                    else:
                        downstream_dis = [downstream_dis[val][0] for val in downstream_dis.keys()]

                    headway_list = [val - pr_dis_dict[pr_vid] for val in downstream_dis]
                    critical_cv[pr_vid] = {"speed": cv_dis_dict[pr_vid] - pr_dis_dict[pr_vid],
                                           "headway": headway_list, "dis": pr_dis_dict[pr_vid]}
                else:
                    critical_cv[pr_vid] = {"speed": cv_dis_dict[pr_vid] - pr_dis_dict[pr_vid],
                                           "headway": [pr_dis_dict[pr_cv_list[vdx + 1]] - pr_dis_dict[pr_vid]],
                                           "dis": pr_dis_dict[pr_vid]}

            # determine the weight of the particles
            if pipeline.particle_weights is None:
                particle_weights = [1 for val in range(particle_number)]
            else:
                particle_weights = pipeline.particle_weights
            particles = pipeline.particles

            for vid in critical_cv.keys():
                speed = critical_cv[vid]["speed"]

                local_particle = particles[vid]
                for pdx in range(particle_number):
                    [locations, lanes_info] = local_particle[pdx]
                    if len(locations) >= 1:
                        headway = locations[0] - critical_cv[vid]["dis"]
                        local_weight = self.car_following.get_weight(headway, speed)

                        if speed < 1 and headway > 11:
                            particles[vid][pdx] = [[critical_cv[vid]["dis"] + 7] + locations, [None] + lanes_info]
                            local_weight = 0.1
                        particle_weights[pdx] *= local_weight
                    else:
                        temp_weight_list = []
                        headway_list = np.sort(critical_cv[vid]["headway"])
                        for headway in headway_list:
                            if speed < 1 and headway > 11:
                                particles[vid][pdx] = [[critical_cv[vid]["dis"] + 7], [None]]
                                temp_weight_list.append(0.1)
                                break

                            temp_weight_list.append(self.car_following.get_weight(headway, speed))
                        particle_weights[pdx] *= np.average(temp_weight_list)

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
            # print("Total weights too small as", total_weights, "invalid particles any more!")
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

        # initiate the pipeline lane change dict
        particle_lane_change_dict = {}              # dict = {"pip_id", [[cut_in_dis,...], [], ]}
        for pip_id in self.pipelines.keys():
            pipeline = self.pipelines[pip_id]
            particle_lane_change_dict[pipeline.id] = [[[], []] for val in range(pipeline.particle_number)]

        particle_lane_change_flag = False

        # Part 1: without lane changing from outside...
        for pip_id in self.pipelines.keys():
            pipeline = self.pipelines[pip_id]
            particles = pipeline.particles
            direction = pipeline.direction

            # initiate the forward particles
            pipeline.forward_particles = [[[], []] for val in range(pipeline.particle_number)]

            # initiate the particle weights
            pipeline.particle_weights = [1 for val in range(pipeline.particle_number)]

            # clear the out flow buffer
            pipeline.outflow = {}

            for i_dir in direction:
                self.pipelines[pip_id].outflow[i_dir] = [None for val in range(pipeline.particle_number)]

            [new_cv_list, new_cv_distances] = pipeline.vehicles
            current_cv_locations_dict = {}
            for idx in range(len(new_cv_list)):
                current_cv_locations_dict[new_cv_list[idx]] = new_cv_distances[idx]

            [cv_list, cv_distance_list] = deepcopy(pipeline.previous_vehicles)
            previous_cv_locations_dict = {}
            for idx in range(len(cv_list)):
                previous_cv_locations_dict[cv_list[idx]] = cv_distance_list[idx]

            downstream_num = len(direction)
            downstream_dis = pipeline.downstream_dis

            particle_keys = list(particles.keys())

            # check cv list
            if particle_keys[1:] != cv_list:
                print("\nreport error information")
                print("Pipeline", pip_id, "not consistent")
                print("Previous vehicles:", pipeline.previous_vehicles)
                print("Current vehicles:", pipeline.vehicles)
                print("Particle keys:", particle_keys)
                exit("Not consistent error!")

            for vdx in range(len(cv_list) + 1):
                if vdx == len(cv_list):
                    mid_flag = False
                    last_distance = self.length
                else:
                    mid_flag = True
                    last_distance = cv_distance_list[vdx]
                cv_id = particle_keys[vdx]
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

                        # already in the destination lane
                        if lane_change == pipeline.index:
                            latest_locations.append(location)
                            latest_lane_infos.append(lane_change)
                            continue

                        # keep in the lane
                        if location < 25:
                            latest_locations.append(location)
                            latest_lane_infos.append(lane_change)
                            continue

                        current_index = pipeline.index
                        dest_pip_id = int(current_index + np.sign(lane_change - current_index))
                        dest_pip_id = pipeline_dict[dest_pip_id]
                        destination_pipeline = self.pipelines[dest_pip_id]
                        start_dis = destination_pipeline.start_dis

                        # keep in the lane
                        if location < start_dis:
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
                                # Finally lane change success...
                                # do not store the current particle but push it to processed in the future
                                particle_lane_change_dict[dest_pip_id][pdx][0].append(location_list[::-1][i_v])
                                particle_lane_change_dict[dest_pip_id][pdx][1].append(lane_change_list[::-1][i_v])
                                particle_lane_change_flag = True

                    if cv_id in current_cv_locations_dict.keys():
                        current_cv_location = current_cv_locations_dict[cv_id]
                        speed = current_cv_location - previous_cv_locations_dict[cv_id]

                        final_locations = latest_locations[::-1]
                        final_lanes = latest_lane_infos[::-1]
                        cut_index = 0
                        for loc_index in range(len(final_locations)):
                            if final_locations[loc_index] - current_cv_location > speed:
                                cut_index = loc_index
                                break
                        final_locations = final_locations[cut_index:]
                        final_lanes = final_lanes[cut_index:]

                        pipeline.particle_weights[pdx] *= pow(0.1, cut_index)
                        pipeline.forward_particles[pdx][0] += final_locations
                        pipeline.forward_particles[pdx][1] += final_lanes
                    else:
                        # pipeline.particles[cv_id][pdx] = [latest_locations[::-1], latest_lane_infos[::-1]]
                        pipeline.forward_particles[pdx][0] += latest_locations[::-1]
                        pipeline.forward_particles[pdx][1] += latest_lane_infos[::-1]

        # deal with the vehicle to be inserted
        if particle_lane_change_flag:
            for pip_id in particle_lane_change_dict.keys():
                pipeline = self.pipelines[pip_id]

                for pdx in range(pipeline.particle_number):
                    # [original_loc, original_lane] = pipeline.particles["start"][pdx]
                    [new_loc, new_lane] = particle_lane_change_dict[pip_id][pdx]
                    # pipeline.particles["start"][pdx] = [new_loc + original_loc, new_lane + original_lane]

                    [original_loc, original_lane] = pipeline.forward_particles[pdx]
                    pipeline.forward_particles[pdx] = [new_loc + original_loc, new_lane + original_lane]

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

        # arrival rate parameter
        self.arrival_rate = 0

        # network topology
        self.length = None
        self.start_dis = None
        self.tail_length = None

        self.exit_length = {}
        self.direction = None
        self.signal = None
        self.movement = None
        self.downstream_links = None
        self.downstream_pipelines = None

        # downstream information and spill over information
        self.downstream_dis = None
        self.outflow = None
        self.upstream_arrival = None
        self.spillover = False
        self.spillover_prob = []

        # real-time state
        self.vehicles = [[], []]
        self.complete_vehicles = [[], []]
        self.previous_vehicles = [[], []]

        # historical eulerian traffic state
        self.density_matrix = []
        self.real_time_density = []
        self.offline_density = []
        self.observed_density = []

        # vehicle trajectory
        self.cv_trajs = {}
        self.non_cv_trajs = {}
        self.particle_history = {}

        # particles
        # start ----- cv1 ----- cv2 ----- ...
        # {"start": [[distance1 < distance2 <...], [dest_pip1, dest_pip2, ...]]}
        self.particles = {"start": [[[], []]] * particle_number}
        self.particle_weights = None
        self.forward_particles = None

    def update_particle_dict(self):
        location_list = [self.forward_particles[pdx][0] for pdx in range(self.particle_number)]
        lane_change_list = [self.forward_particles[pdx][1] for pdx in range(self.particle_number)]

        new_cv_list = ["start"] + self.vehicles[0]
        cv_distance_list = self.vehicles[1] + [1000000]

        # initiate the new particle
        new_particle_dict = {}
        for cv_id in new_cv_list:
            new_particle_dict[cv_id] = [[[], []] for val in range(self.particle_number)]

        for ip in range(self.particle_number):
            local_locations = location_list[ip]
            local_lane_change = lane_change_list[ip]

            # check sequence
            sequence_correct = True
            for idx in range(len(local_locations) - 1):
                if local_locations[idx + 1] < local_locations[idx]:
                    sequence_correct = False
                    break
            if not sequence_correct:
                new_sequence = np.argsort(local_locations)
                correct_locations = []
                correct_lane_changes = []
                for idx in range(len(new_sequence)):
                    correct_locations.append(local_locations[new_sequence[idx]])
                    correct_lane_changes.append(local_lane_change[new_sequence[idx]])
            else:
                correct_locations = local_locations
                correct_lane_changes = local_lane_change

            cursor = 0
            for idx in range(len(correct_locations)):
                local_loc = correct_locations[idx]
                local_lane = correct_lane_changes[idx]
                while local_loc > cv_distance_list[cursor]:
                    cursor += 1
                new_particle_dict[new_cv_list[cursor]][ip][0].append(local_loc)
                new_particle_dict[new_cv_list[cursor]][ip][1].append(local_lane)

        self.particles = new_particle_dict

    def new_cv_arrived(self, cv_id, cv_dis):
        old_keys = list(self.particles.keys())
        new_keys = old_keys[:1] + [cv_id] + old_keys[1:]
        self.previous_vehicles[0] = [cv_id] + self.previous_vehicles[0]
        self.previous_vehicles[1] = [cv_dis] + self.previous_vehicles[1]

        new_particles = {}
        for ik in new_keys:
            if ik == "start":
                new_particles[ik] = [[[], []] for val in range(self.particle_number)]
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
                self.forward_particles[ip][0] = [arrival_list[ip] - 1] + self.forward_particles[ip][0]

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
                    # self.particles["start"][ip][1] = [chosen_pip] + self.particles["start"][ip][1]
                    self.forward_particles[ip][1] = [chosen_pip] + self.forward_particles[ip][1]
                else:
                    # self.particles["start"][ip][1] = [None] + self.particles["start"][ip][1]
                    self.forward_particles[ip][1] = [None] + self.forward_particles[ip][1]
        return 0

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

