from gymnasium.envs.registration import register

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from lights_env import LightsEnv, TIME_LIMIT

if __name__ == "__main__":
    register(
        id='Lights-v1',
        entry_point=LightsEnv,
        max_episode_steps=TIME_LIMIT // 100 + 1,
    )

    models_dir = 'models_lights'
    logs_dir = 'logs_lights'

    num_cpu = 1
    # DummyVecEnv should be faster, because it creates only 1 process w/ multiple envs
    vec_env = make_vec_env('Lights-v1', n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv, env_kwargs={'render_mode': 'human'})

    model = PPO.load(f'{models_dir}/2023_10_30 23_10_28.209729_ppo0_01_fixed_awt_4/42800.zip')
    #model = DQN.load(f'{models_dir}/dqn/100000.zip')

    print('Testing')
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=False)
        #action = [vec_env.action_space.sample()] * num_cpu
        obs, reward, done, info = vec_env.step(action)
        vec_env.render('human')
        if done.any():
            break
    vec_env.close()
