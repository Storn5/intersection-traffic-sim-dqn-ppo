import os
import sys
from datetime import datetime
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from sim_env import SimEnv, TIME_LIMIT

register(
    id='SimCar-v1',
    entry_point=SimEnv,
    max_episode_steps=TIME_LIMIT // 10 + 1,
)

if __name__ == "__main__":
    model_name = 'ppo'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    model_name = f'{datetime.now()}-{model_name}'.replace(':', '_').replace('-', '_')
    ent_coef = 0.001
    if len(sys.argv) > 2:
        ent_coef = float(sys.argv[2])

    models_dir = f'models/{model_name}'
    logs_dir = 'logs'

    for folder in (models_dir, logs_dir):
        if not os.path.exists(folder):
            os.makedirs(folder)

    num_cpu = 8
    # DummyVecEnv should be faster, because it creates only 1 process w/ multiple envs
    vec_env = make_vec_env('SimCar-v1', n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'render_mode': None})#, monitor_dir=f'{logs_dir}/monitor')

    print('Training')
    # Iteration steps = num_cpu * n_steps = 10000
    # num_cpu * n_steps (real batch size) has to be divisible by batch_size (minibatch size)
    model = PPO('MlpPolicy', vec_env, verbose=0, device='cpu', tensorboard_log=logs_dir, n_steps=1250, batch_size=50, ent_coef=ent_coef, learning_rate=0.001)

    i = 0
    while i < 300:
        i += 1
        model.learn(total_timesteps=10_000, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
        model.save(f'{models_dir}/{10_000 * i}')
    vec_env.close()
