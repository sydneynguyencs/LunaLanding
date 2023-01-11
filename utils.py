import os
import uuid
import gymnasium as gym
from gymnasium import Env


def add_recording(env: Env, save_path: str, n_episodes: int) -> Env:
    env = gym.wrappers.RecordVideo(env, save_path,
                                   episode_trigger=lambda x: x == 100 or x == n_episodes - 1)
    return env


def generate_run_id() -> str:
    return str(uuid.uuid4())


def get_paths(root: str, algo: str, params: dict) -> (str, str, str):
    model_path = root + "/" + algo + "/" + str(params) + "/model"
    result_path = root + "/" + algo + "/" + str(params) + "/result"
    video_path = root + "/" + algo + "/" + str(params) + "/video"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    return model_path, result_path, video_path
