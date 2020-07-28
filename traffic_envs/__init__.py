from gym.envs.registration import register

register(id='plymouth-trafficlight-v0',
         entry_point='traffic_envs.traffic_env:SignalizedNetwork')
