import os
import uuid
import gymnasium as gym
import numpy as np
from gymnasium import Env
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def add_recording(env: Env, save_path: str) -> Env:
    env = gym.wrappers.RecordVideo(env, save_path,
                                   episode_trigger=lambda x: x % 50 == 0)
    return env


def generate_run_id() -> str:
    return str(uuid.uuid4())


def get_paths(root: str, algo: str, params: dict) -> (str, str, str):
    params_string = params_to_string(params)
    model_path = root + "/" + algo + "/" + params_string + "/model"
    result_path = root + "/" + algo + "/" + params_string + "/result"
    video_path = root + "/" + algo + "/" + params_string + "/video"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    return model_path, result_path, video_path


def params_to_string(params: dict) -> str:
    return 'epsilon' + str(params['epsilon']) + '-' + 'gamma' + str(params['gamma']) + '-' + 'learning_rate' \
        + str(params['learning_rate'])


def read_scores(path: str) -> pd.DataFrame:
    _scores = pd.read_csv(path, sep=',', names=['Episode', 'Score'], usecols=['Score'])
    _scores = _scores['Score'].str.replace('Score:', '').astype(float)
    _scores = pd.DataFrame(_scores, columns=['Score'])
    return _scores


def plot_scores(_scores: pd.DataFrame, algo_name: str, params: str, save_path: str) -> None:
    _scores.plot()
    x_y_spline = make_interp_spline(_scores.index, _scores['Score'])
    x_ = np.linspace(_scores.index.min(), _scores.index.max(), 20)
    y_ = x_y_spline(x_)
    plt.plot(x_, y_)
    plt.title(algo_name.upper() + "\n" + params)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(save_path)
    plt.show()
