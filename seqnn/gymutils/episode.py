import time
from .agent import RandomAgent


class Episode:
    def __init__(self, env, agent=None):
        if agent is None:
            agent = RandomAgent(env)
        self.env = env
        self.agent = agent

    def run(self, callback=None, sleep=0.0):

        # start new episode
        obs_before, info = self.env.reset()
        done = False
        truncated = False

        while not done and not truncated:

            time.sleep(sleep)

            # choose action
            action = self.agent.get_action()

            # take step in environment
            obs, reward, done, truncated, info = self.env.step(action)
            if callback:
                callback(obs_before, obs, action, reward, done, truncated, info)
            self.agent.update(
                obs_before=obs_before, obs_after=obs, action=action, reward=reward
            )
            obs_before = obs

        self.env.close()
