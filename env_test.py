import numpy as np
try:
    from policy.max_pressure import MaxPressurePolicy as Controller
    print("Using the max pressure controller")
except ImportError:
    from policy.null_policy import NullPolicy as Controller
    print("Using the default actuate controller")
from traffic_envs.traffic_env import SignalizedNetwork


def run_env():
    # make new gym env
    env = SignalizedNetwork()
    # enable output figure
    env.output_cost = True
    # create actor
    actor = Controller(env.action_space)

    # set a random seed for all tests
    env.seed(-1)

    # reset simulation and get new observation
    obs = env.reset()
    action = actor.get_action(obs)
    if action is None:
        env.set_mode(actuate_control=False)
    else:
        env.set_mode(actuate_control=True)
    # with switching cost consideration
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
    # insert the code to be profiled here
    run_env()
