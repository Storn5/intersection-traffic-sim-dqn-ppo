from gymnasium.envs.registration import register

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from sim_env import SimEnv, TIME_LIMIT

if __name__ == "__main__":
    register(
        id='SimCar-v1',
        entry_point=SimEnv,
        max_episode_steps=TIME_LIMIT // 10 + 1,
    )

    models_dir = 'models'
    logs_dir = 'logs'

    num_cpu = 8
    # DummyVecEnv should be faster, because it creates only 1 process w/ multiple envs
    vec_env = make_vec_env('SimCar-v1', n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)

    #model = PPO.load(f'{models_dir}/2023_11_04 17_12_09.493274_ppo0_0001_lights/1630000.zip')
    model = DQN.load(f'{models_dir}/2023_11_04 18_35_30.316007_dqn0_1_lights/1080000.zip')

    print('Testing')
    steps_per_episode = 0
    obs = vec_env.reset()
    for i in range(3000):
        steps_per_episode += 1
        action, _state = model.predict(obs, deterministic=False)
        #action = [vec_env.action_space.sample()] * num_cpu
        obs, reward, done, info = vec_env.step(action)
        if done.any():
            print(reward)
            print(f'Episode over, steps: {steps_per_episode}')
        if i % 10 == 0:
            vec_env.render('human')
    vec_env.close()
