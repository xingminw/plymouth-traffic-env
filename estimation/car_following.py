import numpy as np
from scipy.stats import norm


class SimplifiedModel(object):
    """

    """
    def __init__(self):
        self.free_flow_speed = 15               # free flow speed
        self.jam_density = 7                    # jam density
        self.free_density = 30
        self.maximum_sigma = 1
        # self.minimum_sigma = 0.8

        self.particle_sigma_max = 5
        self.particle_sigma_min = 0.4

    def _get_mean(self, headway):
        """
        get mean and std variance according to distance headway
        :param headway:
        :return:
        """
        if headway < -5:
            return self.free_flow_speed
        if headway <= self.jam_density:
            return 0
        elif headway <= self.free_density:
            proportion = (headway - self.jam_density) / (self.free_density - self.jam_density)
            mean_speed = proportion * self.free_flow_speed
            return mean_speed
        elif headway > self.free_density:
            return self.free_flow_speed
        else:
            print("check the error, the headway is:", headway)

    def _get_disturbed_speed(self, headway):
        mean_speed = self._get_mean(headway)
        if headway <= self.jam_density:
            offset = 0
        elif headway < self.free_density:
            proportion = (headway - self.jam_density) / (self.free_density - self.jam_density)
            variance = proportion * self.maximum_sigma
            offset = norm.rvs(0, variance, 1)[0]
        else:
            offset = norm.rvs(0, self.maximum_sigma, 1)[0]
        return mean_speed + offset

    def _get_speed_list(self, location_list):
        headway_list = np.diff(location_list)
        speed_list = [self._get_disturbed_speed(val) for val in headway_list]
        return speed_list

    def sample_next_locations(self, location_list, _, time_interval):
        """

        :param location_list:
        :param _: null
        :param time_interval:
        :return:
        """
        speed_list = self._get_speed_list(location_list)
        new_locations = np.array(location_list[:-1]) + time_interval * np.array(speed_list)
        return new_locations.tolist()

    def get_weight(self, headway, speed):
        if speed > self.free_flow_speed:
            std_val = self.particle_sigma_max
        elif speed <= 0:
            std_val = self.particle_sigma_min
        else:
            proportion = speed / self.free_flow_speed
            std_val = proportion * self.particle_sigma_max + (1 - proportion) * self.particle_sigma_min
        mean_speed = self._get_mean(headway)
        weight = norm.pdf(speed, mean_speed, std_val) * 100
        # print("Following speed is", speed, "headway is", headway, ", the weight:", weight)
        return weight


class StochasticNewellCarFollowing(object):
    """
    class for a naive stochastic Newell's car-following
    including: how to sample speed from distance headway and
                give the probability distribution according to a headway/speed pair
    """
    def __init__(self):
        self.free_flow_speed = 15
        self.jam_density = 7
        self.free_density = 50
        self.maximum_sigma = 2.5
        self.minimum_sigma = 0.01

    def sample_next_locations(self, location_list, _, time_interval):
        """

        :param location_list:
        :param _: null
        :param time_interval:
        :return:
        """
        speed_list = self._get_speed_list(location_list)
        new_locations = np.array(location_list[1:]) + time_interval * np.array(speed_list)
        return new_locations.tolist()

    def sample_following_speed(self, headway, sample_size):
        """
        sample the following speed according to the distance headway
        :param headway:
        :param sample_size:
        :return:
        """
        [mean, sigma] = self._get_mean_sigma(headway)
        return norm.rvs(mean, sigma, sample_size)

    def get_particle_weight(self):
        pass

    def get_pdf(self, headway, speed):
        [mean, sigma] = self._get_mean_sigma(headway)
        return norm.pdf(speed, mean, sigma)

    def _get_speed_list(self, location_list):
        headway_list = [-val for val in np.diff(location_list)]
        speed_list = [self.sample_following_speed(val, 1)[0] for val in headway_list]
        return speed_list

    def _get_mean_sigma(self, headway):
        """
        get mean and std variance according to distance headway
        :param headway:
        :return:
        """
        if headway <= self.jam_density:
            return [0, self.minimum_sigma]
        elif headway <= self.free_density:
            proportion = (headway - self.jam_density) / (self.free_density - self.jam_density)
            mean_speed = proportion * self.free_flow_speed
            sigma = self.minimum_sigma + (self.maximum_sigma - self.minimum_sigma) * proportion
            return [mean_speed, sigma]
        elif headway > self.free_density:
            return [self.free_flow_speed, self.maximum_sigma]
        else:
            print("check the error, the headway is:", headway)
