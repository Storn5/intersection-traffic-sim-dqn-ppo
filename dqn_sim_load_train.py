import sys
from datetime import datetime
from typing import Callable
from gymnasium.envs.registration import register

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from sim_env import SimEnv, TIME_LIMIT

register(
    id='SimCar-v1',
    entry_point=SimEnv,
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

    models_dir = f'models'
    logs_dir = 'logs'

    num_cpu = 8
    # DummyVecEnv should be faster, because it creates only 1 process w/ multiple envs
    vec_env = make_vec_env('SimCar-v1', n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv, monitor_dir=logs_dir, env_kwargs={'render_mode': None})
    
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
    model = DQN.load(f'{models_dir}/2023_11_03 23_01_54.921969_dqn1_decay_lights/520000.zip')
    model.set_env(vec_env)

    i = 0
    while i < 48:
        i += 1
        model.learn(total_timesteps=10_000, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
        model.save(f'{models_dir}/{10_000 * i}')
    vec_env.close()
