import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import warnings
import dqn
import ddqn
from utils import add_recording, generate_model_id
import numpy as np
import os

warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="LunaLanding")

parser.add_argument("--continuous", type=bool, default=False, help="Continuous Environment")
parser.add_argument(
    "--n_episodes",
    type=int,
    default=500,
    help="Number of episodes.",
)
parser.add_argument(
    "--save_path", type=str, default="experiments", help="Path for model and logs"
)
parser.add_argument(
    "--render_mode", type=str, default="rgb_array", help="Render mode"
)
parser.add_argument(
    "--algorithm", type=str, default="dqn", help="Algorithm"
)
args = parser.parse_args()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():

    args.model_save_path = args.save_path + "/" + args.algorithm + "/model"
    args.result_save_path = args.save_path + "/" + args.algorithm + "/result"
    args.video_save_path = args.save_path + "/" + args.algorithm + "/video"
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)
    os.makedirs(args.video_save_path, exist_ok=True)


    # Set up environment
    env = gym.make(
        "LunarLander-v2",
        continuous=args.continuous,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        render_mode=args.render_mode
    )
    np.random.seed(42)
    env.action_space.seed(42)
    env.reset(seed=42)
    # print(env.observation_space)
    # print(env.action_space)

    # Apply algo
    loss = []
    if args.algorithm == "dqn":
        config = {
            "epsilon": 1.0,
            "gamma": .99,
            "epsilon_min": .01,
            "learning_rate": 0.001,
            "epsilon_decay": .996,
            "memory": 1000000
        }
        model_id = generate_model_id()

        env = add_recording(env=env, algo="dqn", config=config, model_id=model_id)
        loss = dqn.train_dqn(args, env=env, config=config, model_id=model_id)

    elif args.algorithm == "ddqn":
        config = {
            "epsilon": 1.0,
            "gamma": .99,
            "epsilon_min": .01,
            "learning_rate": 0.001,
            "epsilon_decay": .996,
            "memory": 1000000
        }
        model_id = generate_model_id()

        env = add_recording(env=env, algo="ddqn", config=config, model_id=model_id)
        loss = ddqn.train_ddqn(env=env, episode=args.n_episodes, config=config, model_id=model_id)

    elif args.algorithm == "a2c":
        pass

    else:
        print("No such algorithm.")
        exit(-1)

    # Visualize
    plt.plot([i + 1 for i in range(0, len(loss), 2)], loss[::2])
    plt.show()


if __name__ == "__main__":
    main()
