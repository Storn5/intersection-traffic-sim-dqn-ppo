import numpy as np
import gymnasium as gym
import os

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# def make_env(env_id: str, rank: int, render_mode, seed: int = 0):
#     def _init():
#         env = gym.make(env_id, render_mode=render_mode)
#         env.reset(seed=seed + rank)
#         return env
#     set_random_seed(seed)
#     return _init

if __name__ == "__main__":
    models_dir = 'models'
    logs_dir = 'logs'

    num_cpu = 8
    #envs = [make_env('LunarLander-v2', i, 'rgb_array') for i in range(num_cpu)]
    #vec_env = SubprocVecEnv(envs)
    # DummyVecEnv should be faster, because it creates only 1 process w/ multiple envs
    vec_env = make_vec_env('LunarLander-v2', n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)

    model = PPO.load(f'{models_dir}/ppo/300000.zip')
    #model = DQN.load(f'{models_dir}/dqn/100000.zip')

    print('Testing')
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render('human')
    vec_env.close()
