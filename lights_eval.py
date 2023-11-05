import numpy as np
import matplotlib.pyplot as plt
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
    vec_env_fixed_awt = make_vec_env('Lights-v1', n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv, env_kwargs={'render_mode': None, 'fixed_time': True, 'reward_awt': True})
    vec_env_fixed_aql = make_vec_env('Lights-v1', n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv, env_kwargs={'render_mode': None, 'fixed_time': True, 'reward_awt': False})
    vec_env_variable_awt = make_vec_env('Lights-v1', n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv, env_kwargs={'render_mode': None, 'fixed_time': False, 'reward_awt': True})
    vec_env_variable_aql = make_vec_env('Lights-v1', n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv, env_kwargs={'render_mode': None, 'fixed_time': False, 'reward_awt': False})

    models = []
    model_names_ppo_continuous_aql = [
        'models_lights/2023_10_30 17_25_47.120821_ppo1_variable_aql',
        'models_lights/2023_10_30 16_37_53.291735_ppo0_1_variable_aql',
        'models_lights/2023_10_30 16_38_10.525195_ppo0_01_variable_aql',
        'models_lights/2023_10_30 17_32_16.835613_ppo0_001_variable_aql',
    ]
    model_names_ppo_continuous_awt = [
        'models_lights/2023_10_30 11_30_05.545957_ppo1',
        'models_lights/2023_10_29 23_55_55.598651_ppo0_1_variable_2',
        'models_lights/2023_10_29 23_56_34.219178_ppo0_01_variable',
        'models_lights/2023_10_30 11_30_20.302466_ppo0_001',
    ]
    model_names_ppo_fixed_aql = [
        'models_lights/2023_10_31 00_01_18.740875_ppo1_fixed_aql',
        'models_lights/2023_10_30 23_47_01.397512_ppo0_1_fixed_aql',
        'models_lights/2023_10_30 23_47_01.173021_ppo0_01_fixed_aql',
        'models_lights/2023_10_30 23_58_43.195321_ppo0_001_fixed_aql',
    ]
    model_names_ppo_fixed_awt = [
        'models_lights/2023_10_30 23_24_30.317584_ppo1_fixed_awt_4',
        'models_lights/2023_10_30 23_10_28.001897_ppo0_1_fixed_awt_4',
        'models_lights/2023_10_30 23_10_28.209729_ppo0_01_fixed_awt_4',
        'models_lights/2023_10_30 23_24_30.110686_ppo0_001_fixed_awt_4',
    ]
    model_names_dqn_aql = [
        'models_lights/2023_10_31 00_30_17.174415_dqn0_01_decay_fixed_aql',
        'models_lights/2023_10_31 00_30_17.009532_dqn0_001_decay_fixed_aql',
        'models_lights/2023_10_31 00_37_56.936346_dqn0_001_fixed_aql',
        'models_lights/2023_10_31 00_37_57.308361_dqn0_0001_fixed_aql',
    ]
    model_names_dqn_awt = [
        'models_lights/2023_10_31 00_44_22.243309_dqn0_01_decay_fixed_awt',
        'models_lights/2023_10_31 00_44_26.496260_dqn0_001_decay_fixed_awt',
        'models_lights/2023_10_31 00_52_09.543803_dqn0_001_fixed_awt',
        'models_lights/2023_10_31 00_52_16.402805_dqn0_0001_fixed_awt',
    ]
    awt_avgs, aql_avgs = [], []
    for model_name in model_names_ppo_continuous_awt:
        print(model_name)
        model = PPO.load(f'{model_name}/20000.zip')
        awt_sum, aql_sum = 0, 0
        obs = vec_env_variable_awt.reset()
        for _ in range(100):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env_variable_awt.step(action)
            awt_sum += info[0]['awt']
            aql_sum += info[0]['aql']
        awt_avgs.append(awt_sum / 100)
        aql_avgs.append(aql_sum / 100)
    for model_name in model_names_ppo_fixed_awt:
        print(model_name)
        model = PPO.load(f'{model_name}/60000.zip')
        awt_sum, aql_sum = 0, 0
        obs = vec_env_fixed_awt.reset()
        for _ in range(100):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env_fixed_awt.step(action)
            awt_sum += info[0]['awt']
            aql_sum += info[0]['aql']
        awt_avgs.append(awt_sum / 100)
        aql_avgs.append(aql_sum / 100)
    for model_name in model_names_dqn_awt:
        print(model_name)
        model = DQN.load(f'{model_name}/20000.zip')
        awt_sum, aql_sum = 0, 0
        obs = vec_env_fixed_awt.reset()
        for _ in range(100):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env_fixed_awt.step(action)
            awt_sum += info[0]['awt']
            aql_sum += info[0]['aql']
        awt_avgs.append(awt_sum / 100)
        aql_avgs.append(aql_sum / 100)
    for model_name in model_names_ppo_continuous_aql:
        print(model_name)
        model = PPO.load(f'{model_name}/20000.zip')
        awt_sum, aql_sum = 0, 0
        obs = vec_env_variable_aql.reset()
        for _ in range(100):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env_variable_aql.step(action)
            awt_sum += info[0]['awt']
            aql_sum += info[0]['aql']
        awt_avgs.append(awt_sum / 100)
        aql_avgs.append(aql_sum / 100)
    for model_name in model_names_ppo_fixed_aql:
        print(model_name)
        model = PPO.load(f'{model_name}/60000.zip')
        awt_sum, aql_sum = 0, 0
        obs = vec_env_fixed_aql.reset()
        for _ in range(100):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env_fixed_aql.step(action)
            awt_sum += info[0]['awt']
            aql_sum += info[0]['aql']
        awt_avgs.append(awt_sum / 100)
        aql_avgs.append(aql_sum / 100)
    for model_name in model_names_dqn_aql:
        print(model_name)
        model = DQN.load(f'{model_name}/20000.zip')
        awt_sum, aql_sum = 0, 0
        obs = vec_env_fixed_aql.reset()
        for _ in range(100):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env_fixed_aql.step(action)
            awt_sum += info[0]['awt']
            aql_sum += info[0]['aql']
        awt_avgs.append(awt_sum / 100)
        aql_avgs.append(aql_sum / 100)

    # Set the x-axis labels to match the 'x' values
    vec_env_variable_aql.close()
    vec_env_variable_awt.close()
    vec_env_fixed_awt.close()
    vec_env_fixed_aql.close()
    x_labels = [
        'PPO_1_AWT_CONT', 'PPO_0_1_AWT_CONT', 'PPO_0_01_AWT_CONT', 'PPO_0_001_AWT_CONT',
        'PPO_1_AWT', 'PPO_0_1_AWT', 'PPO_0_01_AWT', 'PPO_0_001_AWT',
        'DQN_01_decay_AWT', 'DQN_001_decay_AWT', 'DQN_001_AWT', 'DQN_0001_AWT',
        'PPO_1_AQL_CONT', 'PPO_0_1_AQL_CONT', 'PPO_0_01_AQL_CONT', 'PPO_0_001_AQL_CONT',
        'PPO_1_AQL', 'PPO_0_1_AQL', 'PPO_0_01_AQL', 'PPO_0_001_AQL',
        'DQN_01_decay_AQL', 'DQN_001_decay_AQL', 'DQN_001_AQL', 'DQN_0001_AQL',
    ]
    x_positions = np.arange(len(x_labels))

    # Create bar graphs for 'awt' and 'aql'
    plt.bar(x_positions - 0.125, awt_avgs, 0.25, label='awt', color='blue')
    plt.bar(x_positions + 0.125, aql_avgs, 0.25, label='aql', color='orange')

    # Add labels, legend, and a title
    plt.legend()
    plt.xticks(x_positions, x_labels, rotation=30, ha='right')
    plt.title('AWT and AQL Values for Different X')

    # Show the plot
    plt.show()
