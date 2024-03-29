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
    score_values = _scores.values
    for i in range(len(score_values)):
        val = score_values[i]
        if val < -500 or val > 450:
            _scores.drop([i], axis=0, inplace=True)
    _scores.plot()
    x_y_spline = make_interp_spline(_scores.index, _scores['Score'])
    x_ = np.linspace(_scores.index.min(), _scores.index.max(), 10)
    y_ = x_y_spline(x_)
    plt.plot(x_, y_)
    plt.title(algo_name.upper() + "\n" + params)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(save_path)
    plt.show()


def calculate_mean(_scores: pd.DataFrame, mean_path: str, params: str) -> None:
    mean = _scores['Score'][-100:].mean()
    # Write scores into file
    mean_file = open(mean_path + "/average.txt", "w")
    mean_file.write(f"Average over last 100 episodes: {mean} with episode length {len(_scores)} with params {params} \n")
    mean_file.flush()
    mean_file.close()
