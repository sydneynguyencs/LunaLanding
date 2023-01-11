import uuid
import gymnasium as gym
import json
from gymnasium import Env


def add_recording(env: Env, algo: str, params: dict, model_id: str) -> Env:
    env = gym.wrappers.RecordVideo(env, f'experiments/{algo}/videos/model-{model_id}',
                                   episode_trigger=lambda x: x % 10 == 0,
                                   name_prefix=json.dumps(params))
    return env


def generate_model_id() -> str:
    return str(uuid.uuid4())
