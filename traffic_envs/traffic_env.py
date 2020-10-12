import os
import gym
import json
import numpy as np
import matplotlib.pyplot as plt
import traffic_envs.config as config
import xml.etree.ElementTree as ET

from abc import ABC
from gym import spaces
from glob import glob
from copy import deepcopy
from random import random
from scipy.stats import binom
from traffic_envs.utils import\
    generate_new_route_configuration, delete_buffer_file


__author__ = "Xingmin Wang"


sumoBinary = config.sumoBinary
if config.GUI_MODE:
    import traci
    sumoBinary += "-gui"
else:
    # use libsumo to replace traci to speed up (cannot use gui...)
    try:
        import libsumo as traci
    except ImportError:
        print("libsumo is not installed correctly, use traci instead...")
        import traci


class SignalizedNetwork(gym.Env, ABC):
    def __init__(self, _=None, terminate_steps=3599, signals=None, edges=None, vehicles=None,
                 lanes=None, junctions=None, links=None, resolution=50, arrival_list=None, loaded_list=None,
                 blocked_list=None, total_departures=0, departure_list=None, cost_list=None,
                 cost=0, delay_list=None, long_waiting_penalty=0.3, _use_random_seed=True,
                 _is_open=False, observation_space=None, action_space=None, observation_mapping_details=None,
                 save_trajs=False, observation=None, output_cost=False, sumo_seed=None, long_waiting_clip=5,
                 actuate_control=False, relative_demand=1, observed_cost=0):
        """

        :param terminate_steps: total time slots to terminate the simulation
        :param signals: dict of signal obj
        :param edges: dict of edges
        :param vehicles: dict of vehicles (None currently)
        :param lanes: dict of lanes
        :param junctions: dict of junctions (all the connectors between edges)
        :param links: dict of links (links of the network, each link connects two intersections or entrance/exit)
        :param resolution: used to grid the map, unit: meters
        :param arrival_list: list of total number of arrivals
        :param loaded_list: list of loaded vehicles
        :param departure_list: list of total number of departures
        :param cost_list: list of the cost ("delay" and penalty for lost demand)
        :param cost: current cost
        :param delay_list: list of the delays
        :param long_waiting_penalty: delay^long_waiting to add additional penalty for long waiting time
        :param _use_random_seed: True to use random seed for each epoch
        :param _is_open: state of the simulation, open or closed
        :param observation_space:
        :param action_space:
        :param observation_mapping_details:
        :param save_trajs: set True to save the detailed trajectory for each vehicle, default False
        :param observation:
        :param long_waiting_clip:
        :param actuate_control:
        :param relative_demand:
        """
        # simulation parameters
        self.penetration_rate = 1                           # set to be None to disable the CV environment
        self.output_cost = output_cost                      # true to output the cost curve
        self.terminate_steps = terminate_steps              # total simulation steps
        self.actuate_control = actuate_control              # set True to apply actuate control
        self.relative_demand = relative_demand              # relative demand level, demand times relative demand
        self.save_trajs = save_trajs                        # set True to output the trajectories
        self.visualize_network_topology = False             # flag to visualize the network topology

        # signalized intersection
        self.signals = signals

        # network topology
        self.edges = edges
        self.lanes = lanes
        self.junctions = junctions
        self.links = links

        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_mapping_details = observation_mapping_details
        self.observation = observation

        self.sumo_seed = sumo_seed
        self._use_random_seed = _use_random_seed
        self._long_waiting_penalty = long_waiting_penalty
        self._resolution = resolution
        self._is_open = _is_open

        self._long_waiting_clip = long_waiting_clip

        # buffer data of the traffic state
        # this buffer data must be reset when the simulation is reset
        self._total_departure = total_departures
        if cost_list is None:
            self.cost_list = []
        else:
            self.cost_list = cost_list
        self.cost = cost
        self.observed_cost = observed_cost
        if delay_list is None:
            self.delay_list = []
        else:
            self.delay_list = delay_list
        if arrival_list is None:
            self.arrival_list = []
        else:
            self.arrival_list = arrival_list
        if blocked_list is None:
            self.blocked_list = []
        else:
            self.blocked_list = blocked_list
        if loaded_list is None:
            self.loaded_list = []
        else:
            self.loaded_list = loaded_list
        if departure_list is None:
            self.departure_list = []
        else:
            self.departure_list = departure_list
        if vehicles is None:
            self.vehicles = {}
        else:
            self.vehicles = vehicles

        # load network topology
        self._load_network_topology()

        # initiate the action and observation space, etc.
        self._init_action_observation_space()

        # output the observation mapping
        self._output_observation_mapping_details()

    def step(self, action):
        """

        :param action:
        :return:
        """
        if not self._is_open:
            self._start_new_simulation()

        traci.simulationStep()
        self._implement_new_policy(action)
        self._load_system_state()

        terminate_flag = False
        time_step = traci.simulation.getTime()
        if time_step > self.terminate_steps:
            terminate_flag = True
            self._close_simulator()
        return self.observation, - self.cost / 10000, terminate_flag, {}

    def reset(self):
        """
        reset the simulation and run for one time slot
        :return:
        """
        self._close_simulator()
        if not self._is_open:
            self._start_new_simulation()
        traci.simulationStep()
        self._load_system_state()
        return self.observation

    def seed(self, seed=None):
        """
        select random seed for the simulation

        if seed is None, the env will choose a random seed for each episode
        if seed is a scalar > 0, env will use this random seed and does not change
        if seed is a scalar < 0, env will choose a random seed and then keep using it
        :param seed:
        :return:
        """
        if seed is None:
            self._use_random_seed = True
        else:
            self._use_random_seed = False
            if seed > 0:
                self.sumo_seed = seed
            else:
                self.sumo_seed = self._generate_random_seed()
            print("Using random seed", self.sumo_seed, "...")

    def close(self):
        """

        :return:
        """
        self._close_simulator()

    def set_mode(self, actuate_control):
        self.actuate_control = actuate_control
        self._load_network_topology()

    def output_vehicle_trajectory(self):
        if not os.path.exists(config.output_trajs_folder):
            os.mkdir(config.output_trajs_folder)
        output_folder = os.path.join(config.output_trajs_folder, str(self.sumo_seed))
        os.mkdir(output_folder)

        # output json file
        trajectory_by_link_dict = {}
        for link_id in self.links.keys():
            link = self.links[link_id]
            trajectories = link.trajectories
            trajectory_by_link_dict[link_id] = trajectories

        trajs_json_file = os.path.join(output_folder, "trajs.json")
        with open(trajs_json_file, "w") as temp_file:
            json.dump(trajectory_by_link_dict, temp_file)

        # output spat info
        spat_dict = {}
        for signal_id in self.signals.keys():
            signal = self.signals[signal_id]
            spat_dict[signal_id] = {}
            movements = signal.movements
            for movement_id in movements.keys():
                movement = movements[movement_id]
                movement_state = movement.state_list
                spat_dict[signal_id][movement_id] = movement_state

        spat_json_file = os.path.join(output_folder, "spat.json")
        with open(spat_json_file, "w") as temp_file:
            json.dump(spat_dict, temp_file)

        # output time space diagram
        self._output_corridor_time_space_diagram(config.corridor_w2e,
                                                 os.path.join(output_folder, "w2e_corridor.png"),
                                                 config.intersection_name_list)
        self._output_corridor_time_space_diagram(config.corridor_e2w,
                                                 os.path.join(output_folder, "e2w_corridor.png"),
                                                 config.intersection_name_list[::-1])
        # self._output_link_time_space_diagram(os.path.join(output_folder, "link_ts"))

    def _output_link_time_space_diagram(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

        lane_colors = ["royalblue", "violet", "dimgrey", "orange", "salmon"]

        for link_id in self.links.keys():
            link = self.links[link_id]
            trajectories = link.trajectories
            lane_numbers = link.maximum_lane_number

            plt.figure()
            # create legend
            for idx in range(lane_numbers):
                plt.plot([], [], color=lane_colors[idx], label="Lane "+str(idx))
            plt.plot([], [], "k:", label="Non-CV", lw=1.5)
            plt.plot([], [], "k-", label="CV")
            plt.legend()
            for trip_id in trajectories.keys():
                trip_time = trajectories[trip_id]["time"]
                trip_dis = trajectories[trip_id]["distance"]
                trip_type = trajectories[trip_id]["type"]
                lanes = trajectories[trip_id]["lane"]
                for idx in range(len(trip_time) - 1):
                    lane_index = lanes[idx]
                    dis_list = [trip_dis[idx] + lane_index, trip_dis[idx + 1] + lane_index]
                    if trip_type:
                        plt.plot([trip_time[idx], trip_time[idx + 1]], dis_list,
                                 color=lane_colors[lane_index], lw=1, linestyle="-")
                    else:
                        plt.plot([trip_time[idx], trip_time[idx + 1]], dis_list,
                                 color=lane_colors[lane_index], lw=1.5, linestyle=":")

            # plot the signal timing plan
            edge_list = link.edge_list
            # enter_list
            signal_edge = edge_list[-1]
            edge = self.edges[signal_edge]
            lane_list = edge.lanes_list
            for lane_id in lane_list:
                lane = self.lanes[lane_id]
                if lane.controlled_movement is not None:
                    [signal_id, movement_id] = lane.controlled_movement
                    movement = self.signals[signal_id].movements[movement_id]
                    signal_state_list = movement.state_list

                    for idx in range(len(signal_state_list)):
                        signal_state = signal_state_list[idx]
                        x_list = [idx - 0.5, idx + 0.5]
                        y_list = [link.length + 5 + 4 * int(lane.lane_id[-1])] * 2
                        if signal_state >= 1:
                            plt.plot(x_list, y_list, "g", lw=1.8)
                        else:
                            plt.plot(x_list, y_list, "r", lw=1.8)

            # plot the edge segments
            start_dis = 0
            for edge_id in edge_list:
                edge = self.edges[edge_id]

                plt.plot([0, self.terminate_steps], [start_dis, start_dis], "k--", lw=1, alpha=0.5)
                start_dis += edge.length

            plt.xlabel("Time (s)")
            plt.ylabel("Distance (m)")
            plt.xlim([0, self.terminate_steps])
            plt.ylim([0, link.length + 20])
            plt.show()

    def _output_corridor_time_space_diagram(self, link_list, file_name, signals):
        start_dis = 0
        buffer = 5
        landmark_list = []
        plt.figure(figsize=[10, 10])
        for link_id in link_list:
            link = self.links[link_id]
            link_length = link.length
            enter_edge = link.edge_list[-1]
            edge = self.edges[enter_edge]

            signal_state_list = None
            # plot the signal state
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
                for idx in range(len(signal_state_list)):
                    signal_state = signal_state_list[idx]
                    x_list = [idx, idx + 1]
                    y_list = [start_dis + link.length] * 2
                    if signal_state >= 1:
                        plt.plot(x_list, y_list, "g", lw=1.8)
                    else:
                        plt.plot(x_list, y_list, "r", lw=1.8)

            straight_lane_list = []

            trajectories = link.trajectories
            for trip_id in trajectories.keys():
                trip_time = trajectories[trip_id]["time"]
                trip_dis = trajectories[trip_id]["distance"]
                trip_type = trajectories[trip_id]["type"]
                lane_idx = trajectories[trip_id]["lane"][-1]
                if len(straight_lane_list) > 0:
                    if not (lane_idx in straight_lane_list):
                        continue

                trip_dis = [val + start_dis for val in trip_dis]

                if not trip_type:
                    plt.plot(trip_time, trip_dis, "k", linewidth=0.5, alpha=0.3)
                else:
                    plt.plot(trip_time, trip_dis, "b", linewidth=0.8, alpha=0.5)
            start_dis = start_dis + link_length + buffer
            landmark_list.append(start_dis)
        landmark_list = landmark_list[:-1]
        plt.yticks(landmark_list, signals)
        plt.xlim([0, 3600])
        plt.ylim([0, start_dis])
        # plt.tight_layout()
        plt.xlabel("Time (s)")
        plt.ylabel("Distance")
        plt.savefig(file_name, dpi=300)
        plt.close()

    def output_network_performance(self, additional_name=None):
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Number of vehicles")

        ax1.plot(np.cumsum(self.departure_list), "r", label="total arrivals")
        ax1.plot(np.cumsum(self.arrival_list), "g", label="total departures")
        ax1.plot(self.delay_list, "b", label="system delay")
        ax1.plot(self.blocked_list, label="blocked vehicles")

        ax1.plot(self.loaded_list[:self.terminate_steps], label="loaded vehicles")

        total_cost = np.sum(self.cost_list) / 1000
        total_cost = str(np.round(total_cost))
        ax1.legend()
        ax2 = ax1.twinx()
        ax2.tick_params(axis='y', labelcolor="m")
        ax2.set_ylabel("System cost (1000s) " + total_cost, color="m")
        ax2.plot([val / 1000 for val in self.cost_list], "m", label="system cost", lw=1)

        if not os.path.exists(config.output_figure_folder):
            os.mkdir(config.output_figure_folder)
        if additional_name is not None:
            plt.savefig(config.output_figure_folder + "/" + additional_name + ".png", dpi=200)
        else:
            # output_train_figure_folder = os.path.join(config.output_figure_folder, "train")
            output_train_figure_folder = config.output_figure_folder
            all_existing_figures = glob(output_train_figure_folder + "/*.png")
            if len(all_existing_figures) > 100:
                for figure_name in all_existing_figures[90]:
                    os.remove(figure_name)

            if not os.path.exists(output_train_figure_folder):
                os.mkdir(output_train_figure_folder)
            plt.savefig(output_train_figure_folder + "/" + str(self.sumo_seed) + ".png", dpi=200)

        plt.close()

    @staticmethod
    def _generate_random_seed():
        random_seed = int(random() * 100000)
        return random_seed

    def _close_simulator(self):
        if self._is_open:
            # close simulation
            traci.close()

            # output the figure
            if self.output_cost:
                self.output_network_performance()

            if self.save_trajs:
                self.output_vehicle_trajectory()

            # close simulator
            print("Close simulation with random seeds", self.sumo_seed, "done.")

            # reset the is open flag
            self._is_open = False
        delete_buffer_file(self.sumo_seed)

    def _clear_simulator_parameters(self):
        """
        clear the buffer!!! important, do not forget check this....
        :return:
        """
        self.cost_list = []
        self.delay_list = []
        self.loaded_list = []
        self.departure_list = []
        self.arrival_list = []
        self.vehicles = {}
        self.blocked_list = []
        self._total_departure = 0
        # clear the signal state
        for signal_id in self.signals.keys():
            signal = self.signals[signal_id]
            movements = signal.movements
            for movement_id in movements.keys():
                self.signals[signal_id].movements[movement_id].state_list = []

    def _generate_observation(self):
        link_list = [val[0] for val in self.observation_mapping_details["links"]]
        signal_list = self.observation_mapping_details["signals"]
        observation_dict = {}
        for link_id in link_list:
            link = self.links[link_id]
            density_matrix = link.current_density
            stop_matrix = link.stopped_density
            observation_dict[link_id + "@0"] = density_matrix
            observation_dict[link_id + "@1"] = stop_matrix
            # print(np.shape(density_matrix))

        signal_phases_num = np.cumsum([len(val) for val in self.observation_mapping_details["phases"]])
        signal_phases_num = [0] + signal_phases_num.tolist()

        signal_state = np.zeros(signal_phases_num[-1])
        for signal_idx in range(len(signal_list)):
            signal_id = signal_list[signal_idx]
            signal = self.signals[signal_id]
            if signal.previous_phase is None:
                signal_state[signal_phases_num[signal_idx]] = 1
            else:
                signal_state[signal.previous_phase + signal_phases_num[signal_idx]] = 1
        observation_dict["signals"] = signal_state
        self.observation = observation_dict

    def _start_new_simulation(self):
        """
        create a new demand configuration file and start a new simulation
        reset the _is_open state and the simulation time step
        :return:
        """
        if self._is_open:
            return

        # clear buffer data
        self._clear_simulator_parameters()

        # get new random seed
        if self._use_random_seed:
            self.sumo_seed = self._generate_random_seed()

        # generate new arrival file
        configuration_file_name, loaded_vehicles_list =\
            generate_new_route_configuration(self.sumo_seed, self.relative_demand)
        self.loaded_list = np.cumsum(loaded_vehicles_list)

        print("Start new simulation with random seeds", self.sumo_seed, "...")
        if config.RANDOM_MODE:
            sumo_cmd = [sumoBinary, "-c", configuration_file_name, "--random"]
        else:
            sumo_cmd = [sumoBinary, "-c", configuration_file_name]

        if config.IGNORE_WARNING:
            sumo_cmd.append("--no-warnings")

        if not config.TELEPORT_MODE:
            sumo_cmd += ["--time-to-teleport", str(self.terminate_steps)]

        # start the simulation
        traci.start(sumo_cmd)

        # reset the simulation state flag
        self._is_open = True

    def _init_action_observation_space(self):
        """
        initiate the action space and observation space
        also prepare the mapping relationship for the policy
        :return:
        """
        intersection_indices = tuple(self.signals.keys())

        action_number_list = []
        movement_link_list = []
        agg_phase_state_list = []
        for signal_id in intersection_indices:
            signal = self.signals[signal_id]
            phases = signal.phases
            action_number_list.append(len(phases))
            phases_indices = tuple(signal.phases)
            phase_state_list = []

            for phase_id in phases_indices:
                phase = signal.phases[phase_id]
                phase_state = phase.signal_state
                state_indicator = []
                for i_state in phase_state:
                    if i_state == "g":
                        state_indicator.append(1/2)
                    elif i_state == "G":
                        state_indicator.append(1)
                    else:
                        state_indicator.append(0)
                phase_state_list.append(tuple(state_indicator))

            agg_phase_state_list.append(tuple(phase_state_list))
            movement_links_tuple = []
            for movement_id in range(len(signal.movements)):
                movement = signal.movements[movement_id]
                movement_links_tuple.append((movement.enter_link, movement.exit_link,
                                             int(movement.enter_lane[-1]), movement.turning_ratio,
                                             movement.direction))
            movement_link_list.append(movement_links_tuple)

        links_details_list = []

        observation_dict = {}
        box_ceil_val = int(self._resolution / 5)
        for link_id in self.links.keys():
            link = self.links[link_id]
            local_observation_shape = (link.cell_number, link.maximum_lane_number)
            links_details_list.append((link_id, local_observation_shape))
            observation_dict[link_id + "@0"] = spaces.Box(low=0, high=box_ceil_val, shape=local_observation_shape)
            observation_dict[link_id + "@1"] = spaces.Box(low=0, high=box_ceil_val, shape=local_observation_shape)
        observation_dict["signals"] = spaces.MultiBinary(int(sum(action_number_list)))
        # print(observation_dict["signals"])
        # initiate the action and observation spaces
        self.observation_space = spaces.Dict(observation_dict)
        self.action_space = spaces.MultiDiscrete(action_number_list)

        # get turning weight
        turning_weight_dict = self._generate_turning_weight_matrix()
        # store the observation mapping details
        observation_mapping_dict = {"movements": tuple(movement_link_list),
                                    "links": links_details_list,
                                    "phases": tuple(agg_phase_state_list),
                                    "signals": intersection_indices,
                                    "turning": turning_weight_dict}

        self.observation_mapping_details = observation_mapping_dict

    def _generate_turning_weight_matrix(self):
        # update the turning coefficient
        lane_change_phi = 1.0
        turning_weight_dict = dict()
        for signal_id in self.signals.keys():
            signal = self.signals[signal_id]
            movements = signal.movements

            for movement_id in movements.keys():

                movement = signal.movements[movement_id]
                movement_id = movement.movement_id
                enter_link_id = movement.enter_link
                exit_link = movement.exit_link

                lane_index = int(movement.enter_lane[-1])

                link = self.links[enter_link_id]
                cell_number = link.cell_number
                lane_number = link.maximum_lane_number
                upstream_weight_matrix = np.zeros((cell_number, lane_number))
                turning_ratio = movement.turning_ratio
                lane_change_cell = cell_number - 1.0
                
                for i_lane in range(lane_number):
                    for i_c in range(cell_number):

                        if lane_index == i_lane:
                            if movement.direction == "l" or movement.direction == "L":
                                turning_coefficient = 1
                            else:
                                logit_coefficient = lane_change_phi * (i_c - lane_change_cell)
                                logit_coefficient = np.exp(lane_change_phi * logit_coefficient)
                                logit_coefficient = logit_coefficient / (logit_coefficient + 1)
                                turning_coefficient = turning_ratio + (1 - turning_ratio) * logit_coefficient
                        else:
                            shift_ratio = 1
                            if movement.direction == "l" or movement.direction == "L":
                                shift_ratio = 0.5
                            logit_coefficient = lane_change_phi * (i_c - (
                                    lane_change_cell - pow(shift_ratio * abs(i_lane - lane_index), 2) - 0.8))
                            logit_coefficient = np.exp(- lane_change_phi * logit_coefficient)
                            logit_coefficient = logit_coefficient / (logit_coefficient + 1)
                            if i_lane == lane_number - 1:
                                logit_coefficient = 0
                            turning_coefficient = turning_ratio * logit_coefficient

                        upstream_weight_matrix[i_c, i_lane] = turning_coefficient

                link = self.links[exit_link]
                cell_number = link.cell_number
                lane_number = link.maximum_lane_number
                downstream_weight_matrix = np.zeros((cell_number, lane_number))

                downstream_junction = link.downstream_junction
                junction = self.junctions[downstream_junction]

                # downstream is a intersection
                if junction.junction_id in self.signals.keys():
                    movements = self.signals[junction.junction_id].movements
                    # choose the turning ratio list
                    turning_ratio_list = np.zeros(lane_number)
                    for temp_m_id in movements.keys():
                        movement = movements[temp_m_id]
                        if movement.enter_link == exit_link:
                            lane_id = int(movement.enter_lane[-1])
                            if lane_id >= lane_number:
                                continue
                            turning_ratio_list[lane_id] = movement.turning_ratio
                    for i_lane in range(lane_number):
                        for i_c in range(cell_number):
                            downstream_weight_matrix[i_c, i_lane] = turning_ratio_list[i_lane]
                turning_weight_dict[movement_id] = [upstream_weight_matrix.tolist(),
                                                    downstream_weight_matrix.tolist()]
        return turning_weight_dict

    def _load_system_state(self):
        """
        load the system state
        :return:
        """
        # update the loaded vehicles
        time_step = traci.simulation.getTime()

        # load signal state
        signal_list = self.observation_mapping_details["signals"]
        for signal_id in signal_list:
            if config.GUI_MODE:
                local_state = traci.trafficlight.getRedYellowGreenState(signal_id)
            else:
                local_state = traci.trafficlight_getRedYellowGreenState(signal_id)
            binary_state_list = []
            for state_idx in range(len(local_state)):
                single_state = local_state[state_idx]
                if single_state == "G" or single_state == "g":
                    binary_state = 1
                else:
                    binary_state = 0
                binary_state_list.append(binary_state)
                # movement_id = list(self.signals[signal_id].movements)[state_idx]

                self.signals[signal_id].movements[state_idx].state_list.append(binary_state)
            self.signals[signal_id].observed_state = binary_state_list

        # load useful information from simulation
        current_departure = int(traci.simulation.getDepartedNumber())
        self._total_departure += current_departure
        current_blocked_num = self.loaded_list[int(time_step) - 1] - self._total_departure
        self.blocked_list.append(current_blocked_num)
        self.departure_list.append(current_departure)

        current_arrival = traci.simulation.getArrivedNumber()
        self.arrival_list.append(current_arrival)

        vehicle_list = traci.vehicle.getIDList()

        # get the link vehicle dict (including cv and non-cv)
        link_vehicle_dict = {}

        for vehicle_id in vehicle_list:
            # fetch the state of the vehicle
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
            waiting_time = traci.vehicle.getWaitingTime(vehicle_id)

            lane = self.lanes[vehicle_lane]
            vehicle_edge = lane.edge_id
            edge = self.edges[vehicle_edge]
            vehicle_link = edge.link_id

            edge_pos = traci.vehicle.getLanePosition(vehicle_id)

            link_pos = self._get_edge_distance_from_start_of_link(vehicle_link, vehicle_edge)
            if link_pos is not None:
                link_pos += edge_pos

            # the following code is to dump the data to vehicle class
            if not (vehicle_id in self.vehicles.keys()):
                self.vehicles[vehicle_id] = Vehicle(vehicle_id)
                if random() > self.penetration_rate:
                    self.vehicles[vehicle_id].cv_type = False
                else:
                    self.vehicles[vehicle_id].cv_type = True

            self.vehicles[vehicle_id].speed_list.append(vehicle_speed)
            self.vehicles[vehicle_id].lane_list.append(vehicle_lane)
            self.vehicles[vehicle_id].link_list.append(vehicle_link)
            self.vehicles[vehicle_id].link_pos_list.append(link_pos)
            if vehicle_link is None:
                continue

            if not (vehicle_link in link_vehicle_dict.keys()):
                link_vehicle_dict[vehicle_link] = {"pos": [], "lane": [], "speed": [],
                                                   "delay": 0, "cost": 0, "observed_cost": 0,
                                                   "cv_type": [], "vehicles": []}

            link_vehicle_dict[vehicle_link]["vehicles"].append(vehicle_id)
            link_vehicle_dict[vehicle_link]["pos"].append(link_pos)
            link_vehicle_dict[vehicle_link]["speed"].append(vehicle_speed)
            link_vehicle_dict[vehicle_link]["lane"].append(int(vehicle_lane[-1]))
            cv_type = self.vehicles[vehicle_id].cv_type
            link_vehicle_dict[vehicle_link]["cv_type"].append(cv_type)

            local_delay = waiting_time > 0
            link_vehicle_dict[vehicle_link]["delay"] += local_delay

            # penalize the long continuous waiting time...
            link_vehicle_dict[vehicle_link]["cost"] +=\
                local_delay * np.clip(pow(waiting_time, self._long_waiting_penalty), 1, self._long_waiting_clip)
            if cv_type:
                link_vehicle_dict[vehicle_link]["observed_cost"] += \
                    local_delay * np.clip(pow(waiting_time, self._long_waiting_penalty), 1, self._long_waiting_clip)

            if self.penetration_rate is not None:
                if not (vehicle_id in self.links[vehicle_link].trajectories.keys()):
                    self.links[vehicle_link].trajectories[vehicle_id] =\
                        {"time": [], "speed": [], "distance": [], "lane": [],
                         "type": self.vehicles[vehicle_id].cv_type}
                self.links[vehicle_link].trajectories[vehicle_id]["time"].append(time_step)
                self.links[vehicle_link].trajectories[vehicle_id]["speed"].append(vehicle_speed)
                self.links[vehicle_link].trajectories[vehicle_id]["distance"].append(link_pos)
                self.links[vehicle_link].trajectories[vehicle_id]["lane"].append(int(vehicle_lane[-1]))

        # sort the vehicle within the link and get their leading following relationship (pipeline)
        self._sort_vehicle_within_link(link_vehicle_dict)

        system_delay = current_blocked_num
        system_cost = self._long_waiting_clip * current_blocked_num
        # sample the observed cost
        observed_cost = self._long_waiting_clip * binom.rvs(int(current_blocked_num), self.penetration_rate, size=1)[0]

        # initiate and clean the link density
        for link_id in self.links.keys():
            link = self.links[link_id]
            cell_number = link.cell_number
            lane_number = self.links[link_id].maximum_lane_number
            density_matrix = np.zeros((cell_number, lane_number))
            self.links[link_id].current_density = density_matrix
            self.links[link_id].stopped_density = deepcopy(density_matrix)
            self.links[link_id].observed_density = deepcopy(density_matrix)
            self.links[link_id].observed_stopped = deepcopy(density_matrix)

        # dump the link vehicle to the Euler coordinates
        for link_id in link_vehicle_dict.keys():
            link = self.links[link_id]
            link_length = link.length
            resolution = self._resolution
            cell_number = link.cell_number

            pos_list = link_vehicle_dict[link_id]["pos"]
            cv_type_list = link_vehicle_dict[link_id]["cv_type"]
            lane_list = link_vehicle_dict[link_id]["lane"]
            speed_list = link_vehicle_dict[link_id]["speed"]
            density_matrix = link.current_density
            stopped_density = link.stopped_density
            observed_density = link.observed_density
            observed_stopped = link.observed_stopped

            for idx in range(len(pos_list)):
                current_pos = pos_list[idx]
                cv_type = cv_type_list[idx]
                distance_to_stopbar = link_length - current_pos
                current_lane = lane_list[idx]
                stop_flag = int(speed_list[idx] < 2)

                local_cell_index = cell_number - int(distance_to_stopbar / resolution) - 1

                density_matrix[local_cell_index, current_lane] += 1
                observed_density[local_cell_index, current_lane] += 1

                if cv_type:
                    stopped_density[local_cell_index, current_lane] += stop_flag
                    observed_stopped[local_cell_index, current_lane] += stop_flag
            self.links[link_id].current_density = density_matrix
            self.links[link_id].current_density = stopped_density
            self.links[link_id].observed_stopped = observed_stopped
            self.links[link_id].observed_density = observed_density
            self.links[link_id].current_vehs = np.sum(density_matrix)

            system_cost += link_vehicle_dict[link_id]["cost"]
            observed_cost += link_vehicle_dict[link_id]["observed_cost"]
            system_delay += link_vehicle_dict[link_id]["delay"]

        self.delay_list.append(system_delay)
        self.cost_list.append(system_cost)
        self.cost = system_cost
        self.observed_cost = observed_cost
        self._generate_observation()

    def _sort_vehicle_within_link(self, link_vehicle_dict):
        for link_id in self.links.keys():
            link = self.links[link_id]

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
                    exit("segment not assigned!")

                # get the pipeline id
                pipeline_assigned = False
                for pipeline_id in pipelines:
                    pipeline_lanes = pipelines[pipeline_id]
                    if vehicle_lane in pipeline_lanes:
                        self.vehicles[vehicle_id].pipeline_list.append(pipeline_id)
                        pipeline_assigned = True

                        if not (pipeline_id in pipeline_vehicle_dict.keys()):
                            pipeline_vehicle_dict[pipeline_id] = {"vehicles": [], "dis": []}
                        pipeline_vehicle_dict[pipeline_id]["vehicles"].append(vehicle_id)
                        pipeline_vehicle_dict[pipeline_id]["dis"].append(vehicle_pos)
                if not pipeline_assigned:
                    exit("pipeline not assigned!")

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
                new_vehicle_list = [None] + new_vehicle_list + [None]
                for vdx in range(len(new_vehicle_list) - 2):
                    following_vehicle = new_vehicle_list[vdx]
                    current_vehicle = new_vehicle_list[vdx + 1]
                    leading_vehicle = new_vehicle_list[vdx + 2]
                    leading_dis = new_distance_list[vdx + 2] - new_distance_list[vdx + 1]
                    following_dis = new_distance_list[vdx + 1] - new_distance_list[vdx]
                    self.vehicles[current_vehicle].leading_dis_list.append(leading_dis)
                    self.vehicles[current_vehicle].leading_vehicle_list.append(leading_vehicle)
                    self.vehicles[current_vehicle].following_dis_list.append(following_dis)
                    self.vehicles[current_vehicle].following_vehicle_list.append(following_vehicle)

            # find the leading cv and following cv
            for vehicle_id in vehicle_list:
                vehicle = self.vehicles[vehicle_id]
                time_out_max = 1000                  # set a time out threshold,
                # forward search
                vehicle_cursor = deepcopy(vehicle)
                count = 0
                while True:
                    count += 1
                    if count > time_out_max:
                        exit("Time out error for finding the forward cv")
                    leading_vehicle = vehicle_cursor.leading_vehicle_list[-1]
                    if leading_vehicle is None:
                        self.vehicles[vehicle_id].leading_cv_list.append(leading_vehicle)
                        break
                    leading_type = self.vehicles[leading_vehicle].cv_type
                    if leading_type:
                        self.vehicles[vehicle_id].leading_cv_list.append(leading_vehicle)
                        break
                    vehicle_cursor = deepcopy(self.vehicles[leading_vehicle])

                vehicle_cursor = deepcopy(vehicle)
                count = 0
                while True:
                    count += 1
                    if count > time_out_max:
                        exit("Time out error for finding the backward cv")
                    following_vehicle = vehicle_cursor.following_vehicle_list[-1]
                    if following_vehicle is None:
                        self.vehicles[vehicle_id].following_cv_list.append(following_vehicle)
                        break
                    following_type = self.vehicles[following_vehicle].cv_type
                    if following_type:
                        self.vehicles[vehicle_id].following_cv_list.append(following_vehicle)
                        break
                    vehicle_cursor = deepcopy(self.vehicles[following_vehicle])

    def _add_signal(self, signal):
        if self.signals is None:
            self.signals = {}
        self.signals[signal.signal_id] = signal

    def _add_junction(self, junction):
        if self.junctions is None:
            self.junctions = {}
        self.junctions[junction.junction_id] = junction

    def _add_edge(self, edge):
        if self.edges is None:
            self.edges = {}
        self.edges[edge.edge_id] = edge

    def _add_lane(self, lane):
        if self.lanes is None:
            self.lanes = {}
        self.lanes[lane.lane_id] = lane

    def _add_link(self, link):
        if self.links is None:
            self.links = {}
        self.links[link.link_id] = link

    def _implement_new_policy(self, new_phases):
        """
        Implement new control policy
        :param new_phases:
        :return:
        """
        if new_phases is None:
            return
        new_phases = list(new_phases)
        signal_id_list = self.observation_mapping_details["signals"]

        for idx in range(len(signal_id_list)):
            signal_id = signal_id_list[idx]

            signal = self.signals[signal_id]
            phases = signal.phases

            if signal.current_state is None:
                phase_state = phases[0].signal_state
                traci.trafficlight.setRedYellowGreenState(signal_id, phase_state)
                self.signals[signal_id].current_state = phase_state
                self.signals[signal_id].previous_phase = 0
                continue

            if signal.force_duration > 0:

                signal.force_duration -= 1
                continue

            if ("y" in signal.current_state) and (signal.cross_barrier is True):

                signal.force_duration = signal.clearance_time
                self.signals[signal_id].current_state = phases[0].signal_state
                self.signals[signal_id].cross_barrier = False
                traci.trafficlight.setRedYellowGreenState(signal_id, phases[0].signal_state)
                continue

            desired_phase = signal.desired_phase
            # implement desired phase if there is.
            if desired_phase is not None:
                new_phase_state = phases[desired_phase].signal_state
                traci.trafficlight.setRedYellowGreenState(signal_id, new_phase_state)
                self.signals[signal_id].current_state = new_phase_state
                self.signals[signal_id].previous_phase = desired_phase
                self.signals[signal_id].desired_phase = None
                # this continue means that the new desired phase must be implemented at least for one time slot
                continue
            # find new control policy
            max_pressure_phase = 0

            new_phase_id = new_phases[idx]
            new_state = phases[new_phase_id].signal_state
            if new_phase_id == signal.previous_phase:
                continue

            transition_state = []
            for i_state in range(len(signal.current_state)):
                previous_state_i = signal.current_state[i_state]
                after_state_i = new_state[i_state]
                if (previous_state_i == "G") or (previous_state_i == "g"):
                    if after_state_i == "r":
                        transition_state.append("y")
                        continue
                transition_state.append(previous_state_i)
            transition_state = "".join(transition_state)

            traci.trafficlight.setRedYellowGreenState(signal_id, transition_state)

            self.signals[signal_id].cross_barrier = \
                (phases[signal.previous_phase].barrier != phases[max_pressure_phase].barrier)
            self.signals[signal_id].force_duration = signal.yellow_time
            self.signals[signal_id].current_state = transition_state
            self.signals[signal_id].desired_phase = new_phase_id

    def _output_observation_mapping_details(self):
        if not os.path.exists(config.output_observation_mapping_file):
            with open(config.output_observation_mapping_file, "w") as temp_file:
                json.dump(self.observation_mapping_details, temp_file)

    def _load_network_topology(self):
        """
        load the network topology from the .net.xml file
        Current issue includes ignoring the "dummy" short edges
        :return:
        """
        #  fetch the network topology in .net.xml file
        if self.actuate_control:
            xml_net_tree = ET.ElementTree(file=config.network_topology_folder + "/actuate.xml")
        else:
            xml_net_tree = ET.ElementTree(file=config.network_topology_folder + "/plymouthv8.net.xml")

        root = xml_net_tree.getroot()

        # load the data for "edge" and "lane"
        for subroot in root:
            if subroot.tag != "edge":
                continue
            current_attrib = subroot.attrib
            edge_id = current_attrib["id"]
            edge = Edge(edge_id)
            if "type" in current_attrib:
                edge_type = current_attrib["type"]
                edge.type = edge_type
            else:
                if "function" in current_attrib:
                    edge.type = current_attrib["function"]

            if "from" in current_attrib:
                origin_junction = current_attrib["from"]
                dest_junction = current_attrib["to"]
                edge.upstream_junction = origin_junction
                edge.downstream_junction = dest_junction

            for subsubroot in subroot:
                lane_attrib = subsubroot.attrib
                lane_id = lane_attrib["id"]
                lane = Lane(lane_id, edge_id)
                lane.length = float(lane_attrib["length"])
                lane.upstream_junction = edge.upstream_junction
                lane.downstream_junction = edge.downstream_junction

                edge.lanes_list.append(lane_id)
                shape_list = lane_attrib["shape"].split(" ")
                shape_list = [[float(val.split(",")[0]) for val in shape_list],
                              [float(val.split(",")[1]) for val in shape_list]]
                lane.shape = shape_list
                self._add_lane(lane)

            self._add_edge(edge)

        # load junction
        for subroot in root:
            if subroot.tag != "junction":
                continue
            junction_attrib = subroot.attrib
            junction_id = junction_attrib["id"]
            junction = Junction(junction_id)
            junction.type = junction_attrib["type"]
            junction.enter_lanes = junction_attrib["incLanes"].split(" ")
            junction.exit_lanes = junction_attrib["intLanes"].split(" ")

            junction.location = [float(junction_attrib["x"]), float(junction_attrib["y"])]
            self._add_junction(junction)

        self._update_junction_edge_connection()

        # load traffic light...
        for subroot in root:
            if subroot.tag != "tlLogic":
                continue

            tl_attrib = subroot.attrib
            signal_id = tl_attrib["id"]
            signal = Signal(signal_id)

            phase_num = len(subroot)
            phase_id = 0
            for subsubroot in subroot:
                phase = Phase(signal_id + "+" + str(phase_id))
                phase.signal_id = signal_id
                phase.signal_state = subsubroot.attrib["state"]
                # signal.phases[phase_id] = subsubroot.attrib["state"]
                if 1 <= phase_id < 1 + (phase_num - 1) / 2:
                    phase.barrier = 0
                elif phase_id > (phase_num - 1) / 2:
                    phase.barrier = 1
                signal.phases[phase_id] = phase
                phase_id += 1

            # add barrier state (all red None)
            self._add_signal(signal)

        for subroot in root:
            if subroot.tag != "connection":
                continue
            connection_attrib = subroot.attrib
            if not ("via" in connection_attrib):
                continue

            upstream_edge = connection_attrib["from"]
            downstream_edge = connection_attrib["to"]

            if not (downstream_edge in self.edges[upstream_edge].downstream_edges):
                self.edges[upstream_edge].downstream_edges.append(downstream_edge)

            if not (upstream_edge in self.edges[downstream_edge].upstream_edges):
                self.edges[downstream_edge].upstream_edges.append(upstream_edge)

            upstream_lane = upstream_edge + "_" + connection_attrib["fromLane"]
            downstream_lane = downstream_edge + "_" + connection_attrib["toLane"]
            via_dummy_lane = connection_attrib["via"]
            direction = connection_attrib["dir"]

            connection_details = {"upstream_lane": upstream_lane, "downstream_lane": downstream_lane,
                                  "via": via_dummy_lane, "dir": direction, "tl": None, "link_id": None}

            if "tl" in connection_attrib:
                connection_details["tl"] = connection_attrib["tl"]
                connection_details["link_id"] = connection_attrib["linkIndex"]
                self.signals[connection_details["tl"]].movements[
                    int(connection_details["link_id"])] = connection_details
            self.lanes[upstream_lane].downstream_lanes[downstream_lane] = connection_details
            self.lanes[downstream_lane].upstream_lanes[upstream_lane] = connection_details

        # generate the link and movement for the network
        # currently ignore the "dummy" link
        self._generate_link_movement(debug=False)

        self._generate_link_segment_and_pipeline(debug=False)

        # update the shape and length of the link from the edges
        self._update_link_shape_from_edges()

        # load turning ratio
        self._load_movement_turning_ratio()

        if self.visualize_network_topology:
            self.output_network_topology_illustration()

    def output_network_topology_illustration(self):
        output_folder = "figs"
        for link_id in self.links.keys():
            plt.figure(figsize=[12, 5])
            link = self.links[link_id]

            for internal_link_id in self.links.keys():
                internal_link = self.links[internal_link_id]
                plt.plot(internal_link.shape[0], internal_link.shape[1], "k")
            plt.title(link_id)
            plt.plot(link.shape[0], link.shape[1], linewidth=2, color="red")
            plt.savefig(os.path.join(output_folder, link_id+".png"))
            plt.close()

    def _load_movement_turning_ratio(self):
        """called by the _load_network_topology
        load the turning ratio of the network
        :return:
        """
        turning_ratio = config.turning_ratio
        for signal_id in self.signals.keys():
            if not (signal_id in turning_ratio.keys()):
                continue
            signal = self.signals[signal_id]
            movements = signal.movements
            for movement_id in movements.keys():
                self.signals[signal_id].movements[movement_id].turning_ratio = turning_ratio[signal_id][movement_id]

    def _get_edge_distance_from_start_of_link(self, link_id, edge_id):
        """
        called by _load_network_topology
        :param link_id:
        :param edge_id:
        :return:
        """
        if link_id is None:
            return None
        link = self.links[link_id]
        link_edges = link.edge_list
        if not (edge_id in link_edges):
            print(edge_id + " not in link " + link_id + "... return None")
            return None
        total_distance = 0
        for temp_edge_id in link_edges:
            if edge_id == temp_edge_id:
                return total_distance
            total_distance += self.edges[temp_edge_id].length
        return total_distance

    def _update_link_shape_from_edges(self):
        """

        :return:
        """
        for link_id in self.links.keys():
            link = self.links[link_id]

            link_shape = [[], []]
            link_length = 0
            maximum_lanes = 0
            minimum_lanes = 10

            for edge_id in link.edge_list:
                edge = self.edges[edge_id]
                edge_shape = edge.shape
                link_shape[0] += edge_shape[0]
                link_shape[1] += edge_shape[1]
                link_length += edge.length

                lane_number = len(edge.lanes_list)
                maximum_lanes = max(maximum_lanes, lane_number)
                minimum_lanes = min(minimum_lanes, lane_number)

            self.links[link_id].shape = link_shape
            self.links[link_id].length = link_length
            self.links[link_id].cell_number = int(np.ceil(link_length / self._resolution)) + 1
            self.links[link_id].maximum_lane_number = maximum_lanes
            self.links[link_id].minimum_lane_number = minimum_lanes

    def _update_junction_edge_connection(self):
        for edge_id in self.edges.keys():
            edge = self.edges[edge_id]
            downstream_junction_id = edge.downstream_junction
            upstream_junction_id = edge.upstream_junction
            if downstream_junction_id is not None:
                self.junctions[downstream_junction_id].enter_edges.append(edge_id)
            if upstream_junction_id is not None:
                self.junctions[upstream_junction_id].exit_edges.append(edge_id)

    def _generate_link_movement(self, debug=False):
        # choose lane 0 as the shape of the edge, so does the length
        for edge_id in self.edges.keys():
            edge = self.edges[edge_id]
            lane_id = edge.lanes_list[0]
            lane = self.lanes[lane_id]

            self.edges[edge_id].shape = lane.shape
            self.edges[edge_id].length = lane.length

        junction_type = []
        signalized_intersection = [[], []]
        destination_node = [[], []]
        right_before_left = [[], []]

        intersection_list = []

        for junction_id in self.junctions:
            junction = self.junctions[junction_id]
            j_t = junction.type
            if not (j_t in junction_type):
                junction_type.append(j_t)

            if j_t == "dead_end":
                destination_node[0].append(junction.location[0])
                destination_node[1].append(junction.location[1])

            if j_t == "right_before_left":
                right_before_left[0].append(junction.location[0])
                right_before_left[1].append(junction.location[1])

            if j_t == "traffic_light":
                intersection_list.append(junction_id)
                signalized_intersection[0].append(junction.location[0])
                signalized_intersection[1].append(junction.location[1])

        for signal_id in intersection_list:
            junction = self.junctions[signal_id]
            edges_list = junction.enter_edges

            for edge_id in self.edges.keys():
                edge = self.edges[edge_id]
                downstream_edges = edge.downstream_edges
                if edge.downstream_junction is None:
                    continue
                downstream_junction = self.junctions[edge.downstream_junction]
                if len(downstream_edges) == 0:
                    if downstream_junction.type == "dead_end":
                        edges_list.append(edge_id)

            # generate the link from edges
            for edge_id in edges_list:
                # edge = self.edges[edge_id]
                link_edges_list = [edge_id]
                link = Link()

                # upstream_intersection = None
                while True:
                    current_edge_id = link_edges_list[0]
                    current_edge = self.edges[current_edge_id]

                    upstream_junction_id = current_edge.upstream_junction
                    upstream_junction = self.junctions[upstream_junction_id]

                    if upstream_junction.type == "traffic_light":
                        upstream_intersection = upstream_junction_id
                        break
                    upstream_edges = current_edge.upstream_edges
                    if len(upstream_edges) == 0:
                        upstream_intersection = upstream_junction_id
                        break

                    if len(upstream_edges) > 1:
                        exit("Number of edges are larger than 1.")
                    link_edges_list = upstream_edges + link_edges_list

                link.edge_list = link_edges_list

                if len(link_edges_list) <= 1:
                    link.link_id = link_edges_list[0]
                else:
                    link.link_id = link_edges_list[0] + "+" + link_edges_list[1]

                # add link id to the edges
                for _edge_id in link.edge_list:
                    self.edges[_edge_id].link_id = link.link_id

                link.upstream_junction = upstream_intersection
                link.downstream_junction = signal_id
                self._add_link(link)

        # generate the movement for signal id
        for signal_id in self.signals.keys():
            signal = self.signals[signal_id]
            movements = signal.movements

            for movement_id in movements.keys():
                movement = Movement(signal_id + "_" + str(movement_id))
                enter_lane = movements[movement_id]["upstream_lane"]
                self.lanes[enter_lane].controlled_movement = [signal_id, movement_id]
                exit_lane = movements[movement_id]["downstream_lane"]
                direction = movements[movement_id]["dir"]
                enter_edge = self.lanes[enter_lane].edge_id

                exit_edge = self.lanes[exit_lane].edge_id

                movement.enter_lane = enter_lane
                movement.exit_lane = exit_lane
                movement.direction = direction

                for link_id in self.links.keys():
                    edge_list = self.links[link_id].edge_list
                    if enter_edge in edge_list:
                        movement.enter_link = link_id
                    if exit_edge in edge_list:
                        movement.exit_link = link_id

                if (movement.enter_link is None) or (movement.exit_link is None):
                    # print("enter_edge", enter_edge, "exit_edge", exit_edge)
                    exit(movement.movement_id + " link not recognized")
                self.signals[signal_id].movements[movement_id] = movement

        # # test plot
        if debug:
            plt.figure(figsize=[12, 5.5])
            for lane_id in self.lanes.keys():
                lane = self.lanes[lane_id]
                lane_shape = lane.shape
                plt.plot(lane_shape[0], lane_shape[1], "k-", lw=0.5)

            for edge_id in self.edges.keys():
                edge = self.edges[edge_id]
                edge_shape = edge.shape
                plt.plot(edge_shape[0], edge_shape[1], "b-", lw=0.7)

            for link_id in self.links.keys():
                link = self.links[link_id]
                link_shape = link.shape
                edge_start = link.edge_list[0]
                edge = self.edges[edge_start]
                plt.plot(link_shape[0], link_shape[1], "r-", lw=1)
                plt.plot(edge.shape[0], edge.shape[1], "m-", lw=1)

            for edge_id in self.edges.keys():
                edge = self.edges[edge_id]
                edge_shape = edge.shape
                if edge.link_id is None:
                    plt.plot(edge_shape[0], edge_shape[1], "b-", lw=2)

            plt.plot(destination_node[0], destination_node[1], "c.", label="dead end")
            plt.plot(right_before_left[0], right_before_left[1], "g.", label="right before left")
            plt.plot(signalized_intersection[0], signalized_intersection[1], "r*", label="signals")
            plt.legend()
            plt.show()

    def _generate_link_segment_and_pipeline(self, debug=False):
        for link_id in self.links.keys():
            link = self.links[link_id]
            edge_list = link.edge_list

            segment_edge_list = [[edge_list[0]]]
            pipeline_lane_list = [[val] for val in self.edges[edge_list[0]].lanes_list]
            segment_cursor = 0
            for idx in range(len(edge_list) - 1):
                local_edge_id = edge_list[idx]
                local_edge = self.edges[local_edge_id]
                next_edge_id = edge_list[idx + 1]
                next_edge = self.edges[next_edge_id]

                local_lanes = local_edge.lanes_list
                next_lanes = next_edge.lanes_list
                # print("\nlocal and next lanes:", local_lanes, next_lanes)

                if len(local_lanes) == len(next_lanes):
                    segment_edge_list[segment_cursor].append(next_edge_id)
                    for ipx in range(len(pipeline_lane_list)):
                        lane_id = pipeline_lane_list[ipx][-1][-1]
                        for next_lane in next_lanes:
                            if next_lane[-1] == lane_id:
                                pipeline_lane_list[ipx].append(next_lane)
                elif len(local_lanes) > len(next_lanes):
                    print("some lane end...")
                else:
                    offset = self._get_edge_connection_offset(local_edge_id, next_edge_id)
                    segment_edge_list.append([next_edge_id])
                    segment_cursor += 1

                    matched_downstream_lanes = []
                    for ipx in range(len(pipeline_lane_list)):
                        lane_id = int(pipeline_lane_list[ipx][-1][-1])
                        for next_lane in next_lanes:
                            if int(next_lane[-1]) - offset == lane_id:
                                pipeline_lane_list[ipx].append(next_lane)
                                matched_downstream_lanes.append(next_lane)
                                break
                    for next_lane in next_lanes:
                        if not (next_lane in matched_downstream_lanes):
                            pipeline_lane_list.append([next_lane])

            # plot the debug figure for the pipeline and segment separation
            if debug:
                plt.figure(figsize=[10, 5])
                plt.subplot(121)
                for edge_id in edge_list:
                    edge = self.edges[edge_id]
                    for lane_id in edge.lanes_list:
                        plt.plot(self.lanes[lane_id].shape[0], self.lanes[lane_id].shape[1], "k.-", lw=1)
                for pipeline_lane in pipeline_lane_list:
                    x_shape = []
                    y_shape = []
                    for lane_id in pipeline_lane:
                        lane = self.lanes[lane_id]
                        x_shape += lane.shape[0]
                        y_shape += lane.shape[1]
                    plt.plot(x_shape, y_shape, ".-")
                plt.subplot(122)
                for segment in segment_edge_list:
                    x_shape = []
                    y_shape = []
                    for edge_id in segment:
                        x_shape += self.edges[edge_id].shape[0]
                        y_shape += self.edges[edge_id].shape[1]
                    plt.plot(x_shape, y_shape, ".--")
                plt.show()
                plt.close()

            # convert pipeline and segment to dict
            pipeline_dict = {}
            for idx in range(len(pipeline_lane_list)):
                pipeline_dict[idx] = pipeline_lane_list[idx]
            self.links[link_id].pipelines = pipeline_dict

            segment_dict = {}
            for idx in range(len(segment_edge_list)):
                segment_dict[idx] = segment_edge_list[idx]
            self.links[link_id].segments = segment_dict

    def _get_edge_connection_offset(self, upstream_edge_id, downstream_edge_id):
        """
        for those edges extend to more lanes, find the offset
        :param upstream_edge_id:
        :param downstream_edge_id:
        :return:
        """
        upstream_lanes = self.edges[upstream_edge_id].lanes_list
        downstream_lanes = self.edges[downstream_edge_id].lanes_list

        upper_limit = len(downstream_lanes) - len(upstream_lanes)
        total_distance_list = []
        for idx in range(upper_limit + 1):
            actuate_downstream_lanes = downstream_lanes[idx: idx + len(upstream_lanes)]

            total_distance = 0
            for jdx in range(len(upstream_lanes)):
                [x1, x2] = self.lanes[upstream_lanes[jdx]].shape[0][-2:]
                [y1, y2] = self.lanes[upstream_lanes[jdx]].shape[1][-2:]
                x0 = self.lanes[actuate_downstream_lanes[jdx]].shape[0][0]
                y0 = self.lanes[actuate_downstream_lanes[jdx]].shape[1][0]
                local_dis = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
                local_dis = local_dis / np.sqrt(np.square(y2 - y1) + np.square(x2 - x1))

                local_dis += np.sqrt(np.square(x2 - x0) + np.square(y2 - y0)) * 3
                total_distance += local_dis
            total_distance_list.append(total_distance)
        offset = np.argmin(total_distance_list)
        return offset


class Junction(object):
    def __init__(self, junction_id, category=None,
                 location=None, enter_edges=None, exit_edges=None,
                 enter_lanes=None, exit_lanes=None):
        self.junction_id = junction_id
        self.type = category
        self.location = location
        self.enter_lanes = enter_lanes
        self.exit_lanes = exit_lanes
        if enter_edges is None:
            self.enter_edges = []
        else:
            self.enter_edges = enter_edges
        if exit_edges is None:
            self.exit_edges = []
        else:
            self.exit_edges = exit_edges


class Edge(object):
    def __init__(self, edge_id, category=None, upstream_junction=None, link_id=None,
                 downstream_junction=None, lanes_list=None,
                 upstream_edges=None, downstream_edges=None,
                 shape=None, length=None):
        """

        :param edge_id:
        :param category:
        :param upstream_junction:
        :param link_id:
        :param downstream_junction:
        :param lanes_list:
        :param upstream_edges:
        :param downstream_edges:
        :param shape:
        :param length:
        """
        self.link_id = link_id
        self.edge_id = edge_id
        self.length = length
        self.upstream_junction = upstream_junction
        self.downstream_junction = downstream_junction
        self.type = category

        if upstream_edges is None:
            self.upstream_edges = []
        else:
            self.upstream_edges = upstream_edges

        if downstream_edges is None:
            self.downstream_edges = []
        else:
            self.downstream_edges = downstream_edges
        self.shape = shape

        if lanes_list is None:
            self.lanes_list = []
        else:
            self.lanes_list = lanes_list


class Lane(object):
    def __init__(self, lane_id, edge_id, speed=None, length=None,
                 shape=None, downstream_junction=None, upstream_junction=None,
                 upstream_lanes=None, downstream_lanes=None, controlled_movement=None):
        self.lane_id = lane_id
        self.edge_id = edge_id
        self.speed = speed
        self.length = length
        self.shape = shape
        self.downstream_junction = downstream_junction
        self.upstream_junction = upstream_junction
        self.controlled_movement = controlled_movement
        if upstream_lanes is None:
            self.upstream_lanes = {}
        else:
            self.upstream_lanes = upstream_lanes
        if downstream_lanes is None:
            self.downstream_lanes = {}
        else:
            self.downstream_lanes = downstream_lanes


class Vehicle(object):
    def __init__(self, vehicle_id, speed_list=None, lane_list=None,
                 link_list=None, link_pos_list=None, leading_vehicle_list=None,
                 lane_changing_list=None, leading_dis_list=None, following_dis_list=None,
                 following_vehicle_list=None, cv_type=False,
                 leading_cv_list=None, following_cv_list=None, segment_list=None,
                 pipeline_list=None):
        """

        :param vehicle_id:
        :param speed_list:
        :param lane_list:
        :param link_list:
        :param link_pos_list:
        :param leading_vehicle_list:
        :param lane_changing_list:
        :param leading_dis_list:
        :param following_dis_list:
        :param following_vehicle_list:
        :param cv_type:
        :param leading_cv_list:
        :param following_cv_list:
        :param segment_list:
        :param pipeline_list
        """
        self.vehicle_id = vehicle_id
        self.cv_type = cv_type

        if segment_list is None:
            self.segment_list = []
        else:
            self.segment_list = segment_list
        if pipeline_list is None:
            self.pipeline_list = []
        else:
            self.pipeline_list = pipeline_list
        if leading_dis_list is None:
            self.leading_dis_list = []
        else:
            self.leading_dis_list = leading_dis_list
        if following_dis_list is None:
            self.following_dis_list = []
        else:
            self.following_dis_list = following_dis_list
        if lane_changing_list is None:
            self.lane_changing_list = []
        else:
            self.lane_changing_list = lane_changing_list
        if leading_cv_list is None:
            self.leading_cv_list = []
        else:
            self.leading_cv_list = leading_cv_list
        if following_cv_list is None:
            self.following_cv_list = []
        else:
            self.following_cv_list = following_cv_list
        if following_vehicle_list is None:
            self.following_vehicle_list = []
        else:
            self.following_vehicle_list = following_vehicle_list
        if leading_vehicle_list is None:
            self.leading_vehicle_list = []
        else:
            self.leading_vehicle_list = leading_vehicle_list
        if speed_list is None:
            self.speed_list = []
        else:
            self.speed_list = speed_list
        if lane_list is None:
            self.lane_list = []
        else:
            self.lane_list = lane_list
        if link_list is None:
            self.link_list = []
        else:
            self.link_list = link_list
        if link_pos_list is None:
            self.link_pos_list = []
        else:
            self.link_pos_list = link_pos_list


class Phase(object):
    def __init__(self, phase_id=None, signal_id=None, barrier=None, signal_state=None, pressure=0):
        self.phase_id = phase_id
        self.pressure = pressure
        self.signal_id = signal_id
        self.barrier = barrier
        self.signal_state = signal_state


class Signal(object):
    def __init__(self, signal_id, phases=None, movements=None,
                 current_state=None, force_duration=0,
                 previous_phase=None, desired_phase=None, observed_state=None):
        """
        class of signal timing plan and signalized intersection
        :param signal_id:
        :param phases:
        :param movements:
        :param current_state:
        :param force_duration:
        :param previous_phase:
        :param desired_phase:
        :param observed_state:
        """
        self.previous_phase = previous_phase
        self.desired_phase = desired_phase
        self.signal_id = signal_id
        self.current_state = current_state
        self.force_duration = force_duration
        self.observed_state = observed_state
        self.clearance_time = 2
        self.yellow_time = 3
        if phases is None:
            self.phases = {}
        else:
            self.phases = phases

        if movements is None:
            self.movements = {}
        else:
            self.movements = movements


class Link(object):
    def __init__(self, link_id=None, edge_list=None, length=None, maximum_lane_number=None,
                 minimum_lane_number=None, cell_number=None,
                 upstream_intersection=None, downstream_intersection=None,
                 shape=None, trajectories=None, current_density=None, stopped_density=None,
                 segments=None, current_vehicles=None, pipelines=None, observed_density=None,
                 observed_stopped=None):
        """
        a link is defined as the whole road segment connects two intersections or ingress/exit node
        :param link_id:
        :param edge_list:
        :param length:
        :param maximum_lane_number:
        :param minimum_lane_number:
        :param cell_number: number of cells
        :param upstream_intersection:
        :param downstream_intersection:
        :param shape:
        :param trajectories:
        :param current_density:
        :param segments: [[edge_id_list], [],...]
        :param current_vehicles:
        :param pipelines: [[edge_id_list], [], [], ...]
        :param observed_stopped: observed stop density matrix
        :param observed_density: observed density matrix
        """
        self.maximum_lane_number = maximum_lane_number
        self.minimum_lane_number = minimum_lane_number
        self.cell_number = cell_number
        self.link_id = link_id
        self.upstream_junction = upstream_intersection
        self.downstream_junction = downstream_intersection
        self.shape = shape
        self.length = length
        self.current_density = current_density
        self.stopped_density = stopped_density
        self.observed_density = observed_density
        self.observed_stopped = observed_stopped
        if trajectories is None:
            self.trajectories = {}
        else:
            self.trajectories = trajectories
        if edge_list is None:
            self.edge_list = []
        else:
            self.edge_list = edge_list
        self.segments = segments
        self.pipelines = pipelines
        if current_vehicles is None:
            self.vehicles = []
        else:
            self.vehicles = current_vehicles


class Movement(object):
    def __init__(self, movement_id, intersection_id=None, direction=None,
                 enter_lane=None, exit_lane=None,
                 enter_link=None, exit_link=None, turning_ratio=1/3,
                 upstream_weight=None, downstream_weight=None,
                 turning_coefficient_matrix=None,
                 state_list=None):
        """
        a movement is defined as inflow lane and the corresponding
            outflow lane at a signalized intersection
        :param movement_id:
        :param intersection_id:
        :param direction: "l", "L", "s", "r"
        :param enter_lane:
        :param exit_lane:
        :param enter_link:
        :param exit_link:
        :param turning_ratio:
        :param upstream_weight:
        :param downstream_weight:
        :param turning_coefficient_matrix:
        :param state_list:
        """
        self.intersection_id = intersection_id
        self.movement_id = movement_id
        self.turning_ratio = turning_ratio
        self.direction = direction
        self.enter_lane = enter_lane
        self.exit_lane = exit_lane
        self.enter_link = enter_link
        self.exit_link = exit_link
        self.upstream_weight = upstream_weight
        self.downstream_weight = downstream_weight
        self.turning_coefficient_matrix = turning_coefficient_matrix

        if state_list is None:
            self.state_list = []
        else:
            self.state_list = state_list
