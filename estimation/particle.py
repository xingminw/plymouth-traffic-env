"""
inherit the whole class of traffic env and add a particle filter estimation part
"""

from abc import ABC
from scipy.stats import uniform
from copy import deepcopy
from estimation.car_following import DeterministicSimplifiedModel
from traffic_envs.traffic_env import SignalizedNetwork, Link

import numpy as np
# import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import traffic_envs.config as env_config


class EstimatedNetwork(SignalizedNetwork, ABC):
    def __init__(self):
        SignalizedNetwork.__init__(self)

        # number of particles
        self.particle_number = 10

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
        print(self.time_step)
        self._sort_vehicle_within_link()

        for link_id in self.links.keys():
            self.links[link_id].particle_forward()

    def _generate_observation(self):
        """

        :return:
        """
        pass

    def _add_link_properties(self, demand_dict, turning_dict):
        signalized_junction_list = list(self.signals.keys())
        for link_id in self.links.keys():
            link = self.links[link_id]

            link.__class__ = ParticleLink
            link.add_attributes()

            pipelines = link.pipelines
            self.links[link_id].ingress_lanes = len(self.edges[link.edge_list[0]].lanes_list)
            self.links[link_id].car_following = DeterministicSimplifiedModel()

            # convert the link pipeline to a class
            new_pipeline_dict = {}
            for pip_idx in pipelines.keys():
                lane_list = pipelines[pip_idx]
                pipeline = PipeLine(lane_list[-1], lane_list, self.particle_number)
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
                    self.links[link_id].particle_arrival = (1 - self.penetration_rate) * demand_dict[start_edge]

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
                for t in range(downstream_num):
                    local_movement = movement_list[t]
                    local_signal = signal_list[t]
                    movement = self.signals[local_signal].movements[local_movement]
                    downstream_lane = movement.exit_lane
                    downstream_link = movement.exit_link
                    downstream_pipeline = self.links[downstream_link].get_lane_belonged_pipeline(downstream_lane)
                    downstream_pips.append(downstream_pipeline)
                self.links[link_id].pipelines[pip_id].downstream_pipelines = downstream_pips

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
                # dump the current vehicle to the previous vehicle
                self.links[link_id].pipelines[pipeline_id].previous_vehicles = \
                    self.links[link_id].pipelines[pipeline_id].vehicles
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

                new_distance_list = [0] + new_distance_list + [link.length]

                # update the vehicle in the link pipelines
                cv_list = []
                cv_distance_list = []
                for vid in new_vehicle_list:
                    if self.vehicles[vid].cv_type:
                        cv_list.append(vid)
                        cv_distance_list.append(self.vehicles[vid].link_pos_list[-1])
                self.links[link_id].pipelines[pipeline_id].vehicles = [cv_list, cv_distance_list]
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

        for link_id in self.links.keys():
            self.links[link_id].update_turning_dict()


class ParticleLink(Link):
    def __init__(self):
        super().__init__()

        self.link_type = None
        self.lane_change_events = None

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

        self.left_pipelines = []
        self.right_pipelines = []
        self.through_pipelines = []
        self.turning_info = None
        self.particle_arrival = None

    def particle_forward(self):
        if self.link_type != "source":
            return
        pipelines = self.pipelines
        link_cv_list = []

        # ge the full list
        for pip_id in pipelines.keys():
            pipeline = pipelines[pip_id]
            [pr_cv_list, _] = pipeline.previous_vehicles
            link_cv_list += pr_cv_list

        for pip_id in pipelines.keys():
            pipeline = pipelines[pip_id]

            [pr_cv_list, pr_cv_distance_list] = pipeline.previous_vehicles
            [cv_list, cv_distance_list] = pipeline.vehicles

            # remove exit vehicles
            particle_key_list = list(pipeline.particles.keys())
            if len(particle_key_list) > 1:
                if not (particle_key_list[-1] in link_cv_list):
                    print("remove", particle_key_list[-1], "from", pip_id, "at")
                    del self.pipelines[pip_id].particles[particle_key_list[-1]]

            new_arrival = False
            # event: new arrival
            if len(cv_list) > 0:
                if not (cv_list[0] in link_cv_list):
                    new_arrival = True

            # if there is no new arrival cv, then generate the arrival for particle
            if self.link_type == "source":
                if not new_arrival:
                    self.pipelines[pip_id].generate_new_arrival(self.particle_arrival, self.turning_info)
                else:
                    # update the pipeline particle accordingly
                    self.pipelines[pip_id].new_cv_arrived(cv_list[0], cv_distance_list[0])

        # deal with the lane changing
        self.sort_lane_changing_events()
        if len(self.lane_change_events) > 0:
            print(self.lane_change_events)
            for cv_id in self.lane_change_events.keys():
                [from_pip, to_pip, cv_dis] = self.lane_change_events[cv_id]
                self.pipelines[from_pip].remove_cv(cv_id)
                self.pipelines[to_pip].insert_cv(cv_id, cv_dis)

        # perform a one-slot car following
        self.step()

        # check
        for pip_id in pipelines.keys():
            pipeline = pipelines[pip_id]

            [pr_cv_list, pr_cv_distance_list] = pipeline.previous_vehicles
            [cv_list, cv_distance_list] = pipeline.vehicles
            # if len(cv_list) > 0:
            particle_keys = list(self.pipelines[pip_id].particles.keys())[1:]
            if particle_keys != cv_list:
                print("new particle does not correct", pip_id, particle_keys,
                      self.pipelines[pip_id].previous_vehicles[0], cv_list)

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
        lane_change_event = self.lane_change_events
        lane_change_flag = len(lane_change_event) + 1
        # print()
        if lane_change_event:
            # print(lane_change_event)
            pass

        # stochastic car-following model
        for pip_id in self.pipelines.keys():
            pipeline = self.pipelines[pip_id]
            particles = pipeline.particles
            direction = pipeline.direction

            [cv_list, cv_distance_list] = deepcopy(pipeline.vehicles)
            if lane_change_flag:
                # print(pip_id, cv_list, particles)
                pass
            cv_distance_list += [1000000]
            downstream_num = len(direction)

            for ido in range(downstream_num):
                pass

            count = 0
            # print(particles.keys(), cv_list)
            for follow_id in particles.keys():
                # print(follow_id)
                last_distance = cv_distance_list[count]
                local_particles = particles[follow_id]
                for pdx in range(pipeline.particle_number):
                    [location_list, lane_change_list] = local_particles[pdx]
                    new_location_list = \
                        self.car_following.sample_next_locations(location_list + [last_distance], None, 1)
                    self.pipelines[pip_id].particles[follow_id][pdx][0] = new_location_list
        if lane_change_flag:
            # exit()
            pass

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

        self.direction = None
        self.signal = None
        self.movement = None
        self.downstream_links = None
        self.downstream_pipelines = None
        self.tail_length = None

        # real-time state
        self.vehicles = [[], []]
        self.previous_vehicles = [[], []]

        # particles
        # start ----- cv1 ----- cv2 ----- ...
        # {"start": [[distance1 < distance2 <...], [dest_pip1, dest_pip2, ...]]}
        self.particles = {"start": [[[], []]] * particle_number}

    def new_cv_arrived(self, cv_id, cv_dis):
        old_keys = list(self.particles.keys())
        new_keys = old_keys[:1] + [cv_id] + old_keys[1:]
        self.previous_vehicles[0] = [cv_id] + self.previous_vehicles[0]
        self.previous_vehicles[1] = [cv_dis] + self.previous_vehicles[1]
        print("add", cv_dis, cv_id, self.id)

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

    def generate_new_arrival(self, arrival_rate, turning_info):
        if arrival_rate is None:
            return
        [ratio_list, pip_list] = turning_info

        # generate the arrival series
        arrival_seeds = uniform.rvs(size=self.particle_number)
        arrival_list = [val < arrival_rate for val in arrival_seeds]

        # generate the turning series
        turning_seeds = uniform.rvs(size=self.particle_number)
        pip_seeds = [int(val * 10) for val in turning_seeds]

        for ip in range(self.particle_number):
            if arrival_list[ip]:
                self.particles["start"][ip][0] = [0] + self.particles["start"][ip][0]

                if turning_seeds[ip] < ratio_list[0]:
                    local_pips = [int(val[-1]) for val in pip_list[0]]
                    chosen_pip = local_pips[pip_seeds[ip] % len(local_pips)]
                    direction = "l"
                elif ratio_list[0] < turning_seeds[ip] < (ratio_list[0] + ratio_list[1]):
                    local_pips = [int(val[-1]) for val in pip_list[0]]
                    chosen_pip = local_pips[pip_seeds[ip] % len(local_pips)]
                    direction = "s"
                else:
                    local_pips = [int(val[-1]) for val in pip_list[0]]
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
            # print()
            # print(self.particles.keys())
            # print(self.previous_vehicles, self.vehicles)
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

    @staticmethod
    def merge(particle1, particle2):
        new_particle = []
        for ip in range(len(particle1)):
            new_particle.append([particle1[ip][0] + particle2[ip][0], particle1[ip][1] + particle2[ip][1]])
        return new_particle
