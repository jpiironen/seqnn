import argparse
import pathlib
import gym
from datetime import datetime
from tqdm import tqdm
import seqnn
from seqnn.control.loss import Setpoint
from seqnn.gymutils.logger import Logger
from seqnn.gymutils.episode import Episode
from seqnn.gymutils.agent import MPCAgent


def get_episode(args):
    if args.render:
        env = gym.make(args.env, max_episode_steps=args.max_len, render_mode="human")
    else:
        env = gym.make(
            args.env, max_episode_steps=args.max_len, render_mode="rgb_array"
        )
    if args.model:
        model = seqnn.load(args.model)
        plan_loss = get_plan_loss(args)
        agent = MPCAgent(
            env,
            model,
            plan_loss,
            plan_horizon=args.plan_horizon,
            num_planning_steps=args.planning_steps,
        )
        episode = Episode(env, agent)
    else:
        # use random actions
        episode = Episode(env)
    return episode


def get_plan_loss(args):
    if args.env == "CartPole-v1":
        return Setpoint(
            reference={"obs0": 0.0, "obs1": 0.0, "obs2": 0.0, "obs3": 0.0},
            weights={"obs0": 0.1, "obs1": 1.0},
            end_only=True,
        )
    raise NotImplementedError(f"No goal specified for environment '{args.env}'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Gym environments with manual control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        help="Environment name.",
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Path to the model to use for MPC",
        default=None,
        required=False,
        type=str,
    )
    parser.add_argument(
        "--planning_steps",
        help="Number of planning steps at each state",
        default=5,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--plan_horizon",
        help="Number of time steps the model predicts into the future. If not given, uses the same horizon as what was used during the training.",
        default=None,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--save_dir",
        help="Path (folder) where to save the data from the episodes. If not given, no data will be saved.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_episodes",
        help="Number of episodes to run.",
        default=1,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--max_len",
        help="Maximum number of steps in each episode.",
        default=None,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--sleep",
        help="Sleep time between two frames. Can be used to reduce CPU burden.",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--render",
        action="store_true",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
    )

    args = parser.parse_args()

    if args.save_dir is not None:
        subdir = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        save_dir = pathlib.Path(args.save_dir) / subdir
        logger = Logger(save_dir)
        for i in tqdm(range(args.num_episodes)):
            episode = get_episode(args)
            frames = episode.run(callback=logger.callback, sleep=args.sleep)
            if args.gif:
                logger.save_gif(save_dir / f"episode{i}.gif", frames)
    else:
        for _ in tqdm(range(args.num_episodes)):
            episode = get_episode(args)
            episode.run(sleep=args.sleep)
