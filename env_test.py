import io
import pstats
import cProfile
import numpy as np

from pstats import SortKey
from policy.null_policy import NullPolicy as Controller

print("Using the default actuate controller")
from traffic_envs.traffic_env import SignalizedNetwork

PROFILE_MODE = False


def run_env():
    # make new gym env
    env = SignalizedNetwork()
    print(env.action_space)                 # output action space

    # set the simulation parameters
    env.output_cost = True                  # enable output figure
    env.penetration_rate = 0                # set the penetration rate
    env.save_trajs = True                   # enable output trajectories
    env.relative_demand = 0.5               # set relative demand level
    env.terminate_steps = 199               # set the simulation steps (MAXIMUM: 3599)
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
    total_delay = np.sum(env.delay_list)
    print("Total delay:", total_delay, "total reward:", total_reward, ".")


if __name__ == '__main__':
    if PROFILE_MODE:
        pr = cProfile.Profile()
        pr.enable()
        # ... do something ...
        run_env()
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    else:
        run_env()

