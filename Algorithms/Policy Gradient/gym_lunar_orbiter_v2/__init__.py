from gym.envs.registration import register

register(
    id='LunarOrbiter-v2',
    entry_point='gym_lunar_orbiter_v2.envs:LunarOrbiterV2',
    max_episode_steps=1000,
    reward_threshold=200,
)
