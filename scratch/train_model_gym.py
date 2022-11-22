import numpy as np
import gym
from seqnn import SeqNN, SeqNNConfig
from seqnn.gymutils.logger import Logger


#envname = 'CartPole-v1'
envname = 'LunarLander-v2'

# read the data
dfs = []
for path in Logger.find_all_files(f"data/gym/{envname}", ".json"):
    df = Logger.load_episode_as_df(path)
    dfs.append(df)

# split into training and validation sets
np.random.seed(3429)
valid_idx = np.random.choice(len(dfs), 5, replace=False)
dfs_train = [df for i, df in enumerate(dfs) if i not in valid_idx]
dfs_valid = [df for i, df in enumerate(dfs) if i in valid_idx]

# setup the model config
env = gym.make(envname)
num_act = env.action_space.n
num_obs = env.observation_space.shape[0]

config = SeqNNConfig(
    targets={"obs": [f"obs{i}" for i in range(num_obs)]},
    controls_categorical={"act0": num_act},
    horizon_past=5,
    horizon_future=10,
    model_args={'dropout': 0.1},
    optimizer="SGD",
    optimizer_args={"lr": 0.001, "momentum": 0.9},
    lr_scheduler_args={"gamma": 0.5, "step_size": 2000},
    scaler_args={'diff_future': True},
    #scaler_args={'center_future': True},
    max_grad_norm=30,
)

# create the model and train
model = SeqNN(config)
model.train(dfs_train, dfs_valid, steps=2.5e4, logdir=f"logs/gym/{envname}")
model.save(f"models/gym/{envname}/model4")
