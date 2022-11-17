# SeqNN
Codebase for experimenting with different types of neural network models for sequential data. The main purpose is to support models that can be used for model based reinforcement learning and control. 

NOTE: The codebase is very much work in progress, and hence especially the documentation is very incomplete. Major changes in syntax/functionality are also possible.

<figure>
<img src="https://jpiironen.github.io/material/seqnn/gym/CartPole-v1/episode0.gif" width="400"/>
<figcaption><i> Example: Model predictive control on Gym's CartPole environment</i></figcaption>
</figure>


## Usage

The code snippets below illustrate the usage with a simple of the [CartPole environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) from the Gym library. Even though we use Gym here for demonstration purposes, the modeling is not tied to Gym in any sense, and using data from other sources is completely possible.

### Collect data 

This will collect and save some data from the CartPole environment with random policy.

```
python -m scripts.run_gym \
  --env CartPole-v1 \
  --num_episodes 30 \
  --save_dir data/gym/CartPole-v1/random
```


### Train a model

Train a model of the environment using the collected data.

```
import numpy as np
import gym
from seqnn import SeqNN, SeqNNConfig
from seqnn.gymutils.logger import Logger


envname = 'CartPole-v1'

# read the data
dfs = []
for path in Logger.find_all_files(f"data/gym/{envname}/random", ".json"):
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
    horizon_past=3,
    horizon_future=5,
    optimizer="SGD",
    optimizer_args={"lr": 0.001, "momentum": 0.9},
    lr_scheduler_args={"gamma": 0.5, "step_size": 2000},
    max_grad_norm=100,
)

# create the model and train
model = SeqNN(config)
model.train(dfs_train, dfs_valid, steps=1.5e4)
model.save(f"models/gym/{envname}/model0")
```

### Training logs

The model training is using [Pytorch Lightining](https://pytorch-lightning.readthedocs.io/en/latest/), so the training logs can be accessed using the Tensorboard, for example like this
```
tensorboard --logdir lightning_logs
```

### Run model predictive control

We can visualize the performance of the trained model in action using a command such as
```
python -m scripts.run_gym \
  --env CartPole-v1 \
  --model models/gym/CartPole-v1/model0 \
  --num_episodes 1 \
  --max_len 500 \
  --render
```
