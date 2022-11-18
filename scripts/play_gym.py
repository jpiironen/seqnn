import argparse
import pathlib
import json
import numpy as np
import gym
import gym.utils.play
import pygame
from datetime import datetime
from seqnn.gymutils.logger import Logger


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
        "--max_len",
        help="Maximum number of steps in each episode.",
        default=None,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--fps", help="Frames per second.", default=20, required=False, type=int
    )
    parser.add_argument(
        "--save_dir",
        help="Path (folder) where to save the data from the episodes. If not given, no data will be saved.",
        default=None,
        required=False,
    )
    args = parser.parse_args()

    if args.env == "CartPole-v1":
        keymap = {
            (pygame.K_LEFT,): 0,
            (pygame.K_RIGHT,): 1,
        }
    elif args.env == "LunarLander-v2":
        keymap = {
            (pygame.K_LEFT,): 1,
            (pygame.K_RIGHT,): 3,
            (pygame.K_UP,): 2,
            (pygame.K_DOWN,): 0,
        }
    elif args.env == "Acrobot-v1":
        keymap = {
            #(pygame.K_LEFT,): 1,
            #(pygame.K_RIGHT,): 2,
            #(pygame.K_DOWN,): 0,
            (pygame.K_LEFT,): 2,
            (pygame.K_RIGHT,): 0,
            (pygame.K_DOWN,): 1,
        }
    else:
        raise ValueError(f"No keyboard mapping for environment {args.env}")

    

    env = gym.make(args.env, max_episode_steps=args.max_len, render_mode="rgb_array")
    if args.save_dir is not None:
        subdir = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        save_dir = pathlib.Path(args.save_dir) / subdir
        logger = Logger(save_dir)
        gym.utils.play.play(
            env, keys_to_action=keymap, callback=logger.callback, fps=args.fps
        )
    else:
        gym.utils.play.play(env, keys_to_action=keymap, fps=args.fps)