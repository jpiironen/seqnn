import numpy as np
import pandas as pd
import torch
from seqnn.utils import get_data_sample
from seqnn.control import CategoricalCEMPlanner
from .logger import Logger


class Agent:
    def __init__(self, env):
        self.env = env

    def update(self, obs_before=None, obs_after=None, action=None, reward=None):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError


class FiniteMemoryAgent(Agent):
    def __init__(self, env, memory_length):
        super().__init__(env)
        self.memory_length = memory_length
        self.obs_before = []
        self.obs_after = []
        self.rewards = []
        self.actions = []

    def update(self, obs_before=None, obs_after=None, action=None, reward=None):
        for storage, value in zip(
            [self.obs_before, self.obs_after, self.rewards, self.actions],
            [obs_before, obs_after, reward, action],
        ):
            if len(storage) == self.memory_length:
                # forget the oldest value
                storage.pop(0)
            storage.append(np.array(value).reshape(1, -1))

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_action(self):
        raise NotImplementedError


class RandomAgent(FiniteMemoryAgent):
    def __init__(self, env):
        super().__init__(env, memory_length=1)

    def get_action(self):
        return self.get_random_action()


class MPCAgent(FiniteMemoryAgent):
    def __init__(self, env, model, plan_loss, plan_horizon=None, num_planning_steps=5):
        super().__init__(env, memory_length=model.config.task.horizon_past)
        self.model = model
        self.plan_loss = plan_loss
        self.plan_horizon = plan_horizon if plan_horizon is not None else self.model.config.task.horizon_future
        self.num_planning_steps = num_planning_steps
        self.check_control_config()

    def check_control_config(self):
        assert (
            len(self.model.config.task.controls_cont) == 0
        ), "Optimization of continuous controls not implemented yet"
        assert (
            len(self.model.config.task.controls_cat) > 0
        ), "No controls, cannot perform MPC"

    def get_df(self):
        return Logger.data_as_df(
            np.concatenate(self.obs_before, axis=0),
            np.concatenate(self.obs_after, axis=0),
            np.concatenate(self.actions, axis=0),
            np.concatenate(self.rewards, axis=0),
        )

    def get_action(self):
        if len(self.obs_after) < self.memory_length:
            # not enough observations for the model, so output random action
            return self.get_random_action()
        df_past = self.get_df()
        dataset = self.model.get_dataset(df_past, past_only=True)
        past = get_data_sample(dataset, indices=-1)

        task = self.model.config.task
        tags_to_optimize = list(col for col in df_past.columns if "act" in col)
        # NOTE: version below would be prettier, but difficult to ensure 
        # the correct ordering w.r.t the environment specs!
        # tags_to_optimize = [
        #    tag for group in task.controls_cat for tag in task.grouping[group]
        # ]
        plan = {
            group: torch.zeros(1, self.plan_horizon, len(task.grouping[group]))
            for group in task.controls_cat
        }
        num_categ = {tag: task.num_categories[tag] for tag in tags_to_optimize}
        planner = CategoricalCEMPlanner(
            self.model, self.plan_loss, past, plan, num_categ
        )

        for _ in range(self.num_planning_steps):
            planner.step()

        # output the first action of the planned sequence
        action = (
            self.model.get_tags(plan, tags_to_optimize)[0, 0, :].numpy().astype(int)
        )
        if len(action) == 1:
            return action.item()
        return action
