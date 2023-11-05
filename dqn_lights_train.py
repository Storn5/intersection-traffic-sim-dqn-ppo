import os
import sys
from datetime import datetime
from typing import Callable
from gymnasium.envs.registration import register

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from lights_env import LightsEnv, TIME_LIMIT

register(
    id='Lights-v1',
    entry_point=LightsEnv,
    max_episode_steps=TIME_LIMIT // 10 + 1,
)

if __name__ == "__main__":
    model_name = 'dqn'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    model_name = f'{datetime.now()}-{model_name}'.replace(':', '_').replace('-', '_')
    lr = 0.0001
    if len(sys.argv) > 2:
        lr = float(sys.argv[2])
    decay = False
    if len(sys.argv) > 3:
        decay = True if sys.argv[3] == '1' else False

    models_dir = f'models_lights/{model_name}'
    logs_dir = 'logs_lights'

    for folder in (models_dir, logs_dir):
        if not os.path.exists(folder):
            os.makedirs(folder)

    num_cpu = 1
    # DummyVecEnv should be faster, because it creates only 1 process w/ multiple envs
    vec_env = make_vec_env('Lights-v1', n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'render_mode': None})#, monitor_dir=f'{logs_dir}/monitor')
    
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func

    print('Training')
    # Iteration steps = num_cpu * n_steps = 200
    # num_cpu * n_steps (real batch size) has to be divisible by batch_size (minibatch size)
    model = DQN('MlpPolicy', vec_env, verbose=0, device='cpu', tensorboard_log=logs_dir, learning_rate=linear_schedule(lr) if decay else lr)

    for i in range(1, 101):
        model.learn(total_timesteps=200, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
        model.save(f'{models_dir}/{200 * i}')
    vec_env.close()
