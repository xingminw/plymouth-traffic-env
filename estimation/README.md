# Stochastic Modeling in Particle Filter

## Stochastic car-following behavior

The customized stochastic car-following model should be a class like this:

```python
class CustomizedCarFollowing(object):
    def __init__(self):
        pass

    def sample_next_locations(self, location_list, speed_list, time_interval):
        """

        :param location_list: location list in the form of [veh1, veh2, ..., veh, leading_cv]
                (veh1 < veh2 <... < leading_cv)
        :param speed_list: [speed1, speed2,...,speed, leading_cv_speed]
        :param time_interval: float
        :return new location list in the form of [veh1,..., veh]
        """
        new_location_list = location_list[:-1]          # an example all the vehicle stay still
        return new_location_list
    
    def get_particle_weight(self, location_list, speed_list):
        """
        :param location_list: location list, [following_cv, veh1, veh2,...,leading_cv]
        :param speed_list: speed list, [following_cv, veh1,..., leading_cv]
        :return weight of the particle, float
        """        
        weight = 1                  # an example with weight 1
        return weight
```

It must contain a member function ```sample_next_locations()``` that needs the input of the 
location list and speed list, returns a new location list.

### Default stochastic car-following model

#### Simplified Newell's car-following model with Gaussian Noise

Maybe I can also try the IDM?

## Lane changing location model