"""
MP policy class

"""

import json
import torch
import numpy as np

from torch import nn
from traffic_envs.config import output_observation_mapping_file

with open(output_observation_mapping_file, "r") as temp_file:
    observation_guidance_dict = json.load(temp_file)


class MaxPressureTorchPolicy(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        # choose the activated parameters to be optimized

        self.movement_priority = None
        self.downstream_weight_dict = None
        self.stop_weight_dict = None

        self.ordinary_weight_dict = None
        self.switching_power = None
        self.switching_coefficient = None
        self.coefficient_matrix_dict = None
        self.signal_links = None

        self.pretuned_params = "sumo/params.json"

        # fetch the observation mapping relationship
        self.observation_guidance = observation_guidance_dict
        # number of movements per signal
        self.movements_num = [len(val) for val in observation_guidance_dict["movements"]]

        link_details_dict = dict()
        for link_detail in self.observation_guidance["links"]:
            link_details_dict[link_detail[0]] = link_detail[1]
        self.link_details = link_details_dict

        self.signal_details = self.observation_guidance["signals"]

        # initiate the parameter to be optimized
        self._initiate_parameters()

        # load the tuned parameter
        self.load_parameters(self.pretuned_params)

        # build the indicator matrix for each phase
        self.phase_indicator = self._build_indicator_matrix()

    def forward(self, observation):
        # flatten the observation
        for obs_key in observation.keys():
            observation[obs_key] = observation[obs_key].flatten(start_dim=1).float()

        # fetch details from mapping relationship
        movement_list = self.observation_guidance["movements"]
        turning_weight_dict = self.observation_guidance["turning"]

        # calculate the pressure for all the movement
        movement_pressure = None
        for idx in range(len(self.signal_details)):
            signal_id = self.signal_details[idx]
            for jdx in range(len(movement_list[idx])):
                movement = movement_list[idx][jdx]
                movement_id = signal_id + "_" + str(jdx)

                # extract information from observation mapping details
                upstream_link, downstream_link, lane_idx, turning_ratio, direction = movement

                # initiate the upstream weight
                upstream_lanes_num = self.link_details[upstream_link][1]
                upstream_turning = torch.tensor(turning_weight_dict[movement_id][0]).flatten().unsqueeze(1)

                upstream_density = observation[upstream_link + "@0"]
                upstream_weight = self.ordinary_weight_dict[upstream_link].unsqueeze(1)
                upstream_weight = torch.mm(upstream_weight,
                                           torch.ones((1, upstream_lanes_num))).flatten().unsqueeze(1)
                upstream_weight = torch.mul(upstream_weight, upstream_turning)
                upstream_weight = torch.mm(upstream_density, upstream_weight)

                upstream_stop_density = observation[upstream_link + "@1"]
                upstream_stop_weight = self.stop_weight_dict[upstream_link].unsqueeze(1)
                upstream_stop_weight = torch.mm(upstream_stop_weight,
                                                torch.ones((1, upstream_lanes_num))).flatten().unsqueeze(1)
                upstream_stop_weight = torch.mul(upstream_stop_weight, upstream_turning)
                upstream_stop_weight = torch.mm(upstream_stop_density, upstream_stop_weight)
                upstream_weight += upstream_stop_weight

                # initiate the downstream pressure
                downstream_lane_number = self.link_details[downstream_link][1]
                downstream_turning = torch.tensor(turning_weight_dict[movement_id][1]).flatten().unsqueeze(1)
                downstream_density = observation[downstream_link + "@1"]
                downstream_weight = self.downstream_weight_dict[downstream_link].unsqueeze(1)
                downstream_weight = torch.mm(downstream_weight,
                                             torch.ones((1, downstream_lane_number))).flatten().unsqueeze(1)
                downstream_weight = torch.mul(downstream_weight, downstream_turning)

                downstream_weight = torch.mm(downstream_density, downstream_weight)
                movement_weight = upstream_weight - downstream_weight
                if movement_pressure is None:
                    movement_pressure = movement_weight
                else:
                    movement_pressure = torch.cat((movement_pressure, movement_weight), dim=1)

        # add priority to each movement
        movement_priority = torch.diag(self.movement_priority)
        movement_pressure = torch.mm(movement_pressure, movement_priority)

        phase_pressure = torch.mm(movement_pressure, self.phase_indicator)

        # deal with the switching barrier engergy cost
        signal_state_list = observation["signals"]

        vehicle_number_list = self._get_switching_cost(observation)

        switching_cost = torch.mul(vehicle_number_list, signal_state_list)

        phase_pressure += switching_cost
        # print(phase_pressure)
        return phase_pressure

    def load_parameters(self, file_name=None):
        if file_name is None:
            return
        with open(file_name, "r") as param_file:
            params_dict = json.load(param_file)
        for layer1 in params_dict.keys():
            content1 = params_dict[layer1]
            if type(content1) == list:
                self.__setattr__(layer1, nn.Parameter(torch.tensor(content1).float()))
            else:
                class_layer2 = self.__getattr__(layer1)
                for layer2 in content1.keys():
                    class_layer2.__setattr__(layer2, nn.Parameter(torch.tensor(content1[layer2]).flatten()))

    def _get_switching_cost(self, observation):
        """
        return the total number of vehicles of each signal
            return value: m * n, m: batch, n: number of vehicles of each intersection
        :param observation:
        :return:
        """
        signal_link_list = self.signal_links
        overall_vehicle_number = None
        for idx in range(len(signal_link_list)):
            links = signal_link_list[idx]
            vehicle_number = None
            for link in links:
                if vehicle_number is None:
                    vehicle_number = torch.sum(observation[link + "@0"], 1)
                else:
                    vehicle_number += torch.sum(observation[link + "@0"], 1)

            number_of_phases = len(self.observation_guidance["phases"][idx])

            # copy vehicle number for each phase
            vehicle_number = torch.pow(vehicle_number, self.switching_power[idx])\
                             * self.switching_coefficient[idx]
            vehicle_number = vehicle_number.unsqueeze(1)
            vehicle_number = torch.mm(vehicle_number, torch.ones(1, number_of_phases))

            if overall_vehicle_number is None:
                overall_vehicle_number = vehicle_number
            else:
                overall_vehicle_number =\
                    torch.cat([overall_vehicle_number, vehicle_number], dim=1)
        return overall_vehicle_number

    def _initiate_parameters(self):
        """
        initiate the parameters to be optimized...

        Max pressure used a pre-determined actor model, so here I did not use the modules in torch.nn
        :return:
        """
        # create weight
        link_list = self.observation_guidance["links"]

        self.ordinary_weight_dict = nn.ParameterDict()
        self.stop_weight_dict = nn.ParameterDict()
        self.downstream_weight_dict = nn.ParameterDict()

        link_shape_dict = dict()
        for link_detail in link_list:
            link_id, (cells, lanes) = link_detail
            link_shape_dict[link_id] = (cells, lanes)

            stop_weight_curve = [pow((cells - idx + 1) / cells, 0.0) / pow(cells, 0.8) for idx in range(cells)]
            # exit()
            stop_initiate_weight = nn.Parameter(torch.tensor(stop_weight_curve, dtype=torch.float32))
            self.stop_weight_dict[link_id] = stop_initiate_weight

            non_stop_weight_curve = [pow((idx + 1) / cells, 1.5) / pow(cells, 0.8) for idx in range(cells)]
            non_stop_initiate_weight = nn.Parameter(torch.tensor(non_stop_weight_curve, dtype=torch.float32))
            self.ordinary_weight_dict[link_id] = non_stop_initiate_weight

            downstream_weight_curve = [pow((cells - idx) / cells, 1.5) / pow(cells, 0.8) for idx in range(cells)]
            downstream_weight_param = nn.Parameter(torch.tensor(downstream_weight_curve, dtype=torch.float32))
            self.downstream_weight_dict[link_id] = downstream_weight_param

        # parameters of the switching
        signal_list = self.observation_guidance["signals"]

        self.switching_coefficient = nn.Parameter(torch.ones(len(signal_list), ))
        self.switching_power = nn.Parameter(torch.ones(len(signal_list), ) * 0.25)

        # parameters of the priority
        movement_list = self.observation_guidance["movements"]
        movement_num_list = [len(val) for val in movement_list]
        # print(movement_num_list)
        total_movements_num = int(sum(movement_num_list))
        # print(signal_list)
        self.movement_priority = nn.Parameter(torch.ones(total_movements_num, ))

        # turning coefficient matrix dict of each movement
        self.coefficient_matrix_dict = self.observation_guidance["turning"]

        link_list_of_signal = []
        for signal_movements in movement_list:
            enter_link_list = []
            for movement in signal_movements:
                upstream_link, _, _, _, _ = movement
                if not (upstream_link in enter_link_list):
                    enter_link_list.append(upstream_link)
            link_list_of_signal.append(enter_link_list)
        self.signal_links = link_list_of_signal

    def _build_indicator_matrix(self):
        phases_details = self.observation_guidance["phases"]

        cum_movements_num = [0] + [int(val) for val in np.cumsum(self.movements_num).tolist()]

        indicator_matrix = []
        total_movements = cum_movements_num[-1]
        for idx in range(len(phases_details)):
            signal_phases = phases_details[idx]

            for phase_id in range(len(signal_phases)):
                before_zeros = [0] * cum_movements_num[idx]
                after_zeros = [0] * (total_movements - cum_movements_num[idx + 1])
                phase = signal_phases[phase_id]
                phase_indicator = before_zeros + phase + after_zeros
                indicator_matrix.append(phase_indicator)
        indicator_matrix = torch.tensor(indicator_matrix).float()
        return indicator_matrix.t()


class MaxPressurePolicy(MaxPressureTorchPolicy):
    def __init__(self, action_space):
        self.action_space = action_space
        MaxPressureTorchPolicy.__init__(self)

    def set_mode(self, switching=True):
        if switching:
            self.load_parameters(self.pretuned_params)
        else:
            # set the switching coefficient to 0
            params_shape = self.switching_coefficient.shape
            self.switching_coefficient = nn.Parameter(torch.zeros(params_shape))

    def get_action(self, observation):
        new_observation_dict = dict()
        for obs_key in observation.keys():
            new_observation_dict[obs_key] = torch.tensor([observation[obs_key]])
        phase_weight_list = self.forward(new_observation_dict)[0]

        phase_separate_list = self.action_space.nvec
        phase_weights = [phase_weight_list[: phase_separate_list[0]]]

        cursor_start = 0
        cursor_end = phase_separate_list[0]
        for idx in range(len(phase_separate_list) - 1):
            current_phases = phase_separate_list[idx]
            next_phases = phase_separate_list[idx + 1]
            cursor_start += current_phases
            cursor_end += next_phases
            phase_weights.append(phase_weight_list[cursor_start: cursor_end])

        action_list = []
        for phase_weight in phase_weights:
            phase_id = torch.argmax(phase_weight)
            action_list.append(int(phase_id))
        return action_list
