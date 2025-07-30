import gymnasium as gym
from PIL import Image
import numpy as np
from PPO import *
from gymnasium.wrappers import RecordVideo,RenderCollection,HumanRendering
import highway_env  # noqa: F401
import cv2
import os
from MyEnv import *
from copy import deepcopy
import yaml
import argparse
import csv
import pandas as pd


env = gym.make("MyEnv-v0", render_mode='rgb_array')
# Run the trained model and record video
model = PPO.load("highway_larl/model", env=env)
env = RecordVideo(
    env, video_folder="highway_larl/videos", episode_trigger=lambda e: True
)
env.unwrapped.set_record_video_wrapper(env)
env.unwrapped.config["duration"] = 60

for videos in range(10):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # Render
        env.render()
env.close()

