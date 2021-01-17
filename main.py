import numpy as np

from traffic_envs.traffic_env import SignalizedNetwork
from policy.null_policy import NullPolicy as Controller


def run_env():
    # make new gym env
    env = SignalizedNetwork()
    print(env.action_space)                 # output action space

    # set the simulation parameters
    env.output_cost = False                  # enable output figure
    env.penetration_rate = 1                # set the penetration rate
    env.save_trajs = False                   # enable output trajectories
    env.relative_demand = 1               # set relative demand level
    env.terminate_steps = 120               # set the simulation steps (MAXIMUM: 3599)
    env.set_mode(actuate_control=True)      # set the controller to be an actuate control
    env.seed(-1)                            # set a random seed for all tests

    actor = Controller(env.action_space)    # create actor
    obs = env.reset()                       # reset simulation and get new observation

    total_reward = 0
    while True:
        action = actor.get_action(obs)
        obs, reward, terminate, _ = env.step(action)
        total_reward += reward
        if terminate:
            break
    average_delay = np.average(env.delay_list)
    print("Average delay:", average_delay, "total reward:", total_reward, ".")


if __name__ == '__main__':
    run_env()
