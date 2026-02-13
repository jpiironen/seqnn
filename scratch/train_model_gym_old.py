import pathlib
import json
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt

from seqnn import SeqNN, SeqNNConfig
from seqnn.gymutils.logger import Logger


envname = 'LunarLander-v2'
#envname = "Acrobot-v1"

dfs = []
for path in Logger.find_all_files(f"data/gym/{envname}/random", ".json"):
    df = Logger.load_episode_as_df(path)

    #####
    obs_names = [col for col in df.columns if 'obs' in col]
    obs_diff_names = [name + '_diff' for name in obs_names]
    df.loc[:,obs_diff_names] = df.loc[:,obs_names].diff().fillna(value=0.0).rename(columns={n1:n2 for n1,n2 in zip(obs_names, obs_diff_names)})
    ####

    dfs.append(df)

np.random.seed(3429)
valid_idx = np.random.choice(len(dfs), 3, replace=False)

dfs_train = [df for i, df in enumerate(dfs) if i not in valid_idx]
dfs_valid = [df for i, df in enumerate(dfs) if i in valid_idx]




env = gym.make(envname)
# TODO: these do not work generally
num_act = env.action_space.n
num_obs = env.observation_space.shape[0]

config = SeqNNConfig(
    targets={"obs": [f"obs{i}" for i in range(num_obs)]},
    controls_categorical={"act0": num_act},
    horizon_past=10,
    horizon_future=15,
    #model='AutoregressiveMLP',
    model_args = {'diff_model': True},
    # model_args={'dropout': 0.1, "num_hidden": [1024,512]},
    #model_args=dict(
    #    num_hidden_dynamics=[256],
    #    num_hidden_readout=[256],
    #    latent_size=128,
    #),
    optimizer="SGD",
    optimizer_args={"lr": 0.001, "momentum": 0.9},
    lr_scheduler_args={"gamma": 0.5, "step_size": 4000},
    max_grad_norm=100,
)

model = SeqNN(config)

model.train(dfs_train, dfs_valid, steps=1e4, dev_run=True)
#model.save(f"models/gym/{envname}/model_diff")
