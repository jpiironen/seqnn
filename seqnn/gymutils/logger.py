import json
import pathlib
import numpy as np
import pandas as pd


class Logger:
    def __init__(self, save_dir):
        self.data = []
        self.episode = 0
        self.save_dir = pathlib.Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

    def callback(
        self,
        obs_before,
        obs_after,
        action,
        reward,
        done,
        truncated,
        info,
    ):
        self.data.append(
            {
                "obs_before": obs_before,
                "obs_after": obs_after,
                "action": action,
                "reward": reward,
            }
        )
        if done or truncated:
            # save and reset
            self.data = self.stack_observations(self.data)
            self.save(self.save_dir / f"episode{self.episode}.json")
            self.data = []
            self.episode += 1

    def stack_observations(self, data):
        assert isinstance(data, list)
        data_stacked = {}
        for key in data[0]:
            data_stacked[key] = []
        for key in data_stacked:
            for obs in data:
                data_stacked[key].append(obs[key])
            data_stacked[key] = np.stack(data_stacked[key])
            n = data_stacked[key].shape[0]
            data_stacked[key] = data_stacked[key].reshape(n, -1)
        return data_stacked

    def save(self, path):
        path = pathlib.Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, "w") as file:
            data = {key: value.tolist() for key, value in self.data.items()}
            json.dump(data, file)

    @staticmethod
    def data_as_df(
        obs_before: np.ndarray,
        obs_after: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ):
        df_act = pd.DataFrame(actions, columns=[f"act{i}" for i in range(actions.shape[1])])
        df_obs = pd.DataFrame(obs_after, columns=[f"obs{i}" for i in range(obs_after.shape[1])])
        df_rew = pd.DataFrame(rewards, columns=["reward"])
        df = pd.concat((df_act, df_obs, df_rew), axis=1)
        return df

    @staticmethod
    def load_json(path):
        with open(path, "rb") as file:
            data = json.load(file)
        data = {key: np.array(value) for key, value in data.items()}
        return data

    @staticmethod
    def load_episode_as_df(path):
        data = Logger.load_json(path)
        return Logger.data_as_df(
            data["obs_before"], data["obs_after"], data["action"], data["reward"]
        )

    @staticmethod
    def find_all_files(path, suffix):
        path = pathlib.Path(path)
        filepaths = []
        for subpath in sorted(list(path.iterdir())):
            if subpath.suffix == suffix:
                filepaths.append(subpath)
            else:
                try:
                    filepaths += Logger.find_all_files(subpath, suffix=suffix)
                except NotADirectoryError:
                    pass
        return filepaths
