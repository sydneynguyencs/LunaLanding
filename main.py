import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import warnings
import dqn
import ddqn
import a2c
from utils import add_recording, get_paths
import numpy as np
from sklearn.model_selection import ParameterGrid

warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="LunarLanding")

parser.add_argument("--continuous", type=bool, default=False, help="Continuous Environment")
parser.add_argument(
    "--n_episodes",
    type=int,
    default=400,
    help="Number of episodes.",
)
parser.add_argument(
    "--save_path", type=str, default="experiments", help="Path for model and logs"
)
parser.add_argument(
    "--render_mode", type=str, default="rgb_array", help="Render mode"
)
parser.add_argument(
    "--algorithm", type=str, default="a2c", help="Algorithm"
)
args = parser.parse_args()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====

def main():
    # Set up environment
    env = gym.make(
        "LunarLander-v2",
        continuous=args.continuous,
        render_mode=args.render_mode
    )
    np.random.seed(42)
    env.action_space.seed(42)
    env.reset(seed=42)
    # print(env.observation_space)
    # print(env.action_space)
    stack = []
    mean_over_last_100_list = []
    params_list = []

    # Apply algo
    if args.algorithm == "dqn":
        # define parameter grid to try out on algorithm
        # params__1 = {'epsilon': 1.0, 'gamma': .99, 'learning_rate': 0.001}
        #params__2 = {'epsilon': 0.5, 'gamma': .99, 'learning_rate': 0.001}
        params__3 = {'epsilon': 1.0, 'gamma': .66, 'learning_rate': 0.001}

        param_list = [params__3]

        # execute on each parameter combination
        
        for params in param_list:
            args.model_save_path, args.result_save_path, args.video_save_path = get_paths(root=args.save_path,
                                                                                          algo=args.algorithm,
                                                                                          params=params)
            # add recording wrapper
            env = add_recording(env=env, save_path=args.video_save_path)
            loss, mean_over_last_100 = dqn.train_dqn(args, env=env, params=params)
            params_list.append(params)
            mean_over_last_100_list.append(mean_over_last_100)

    elif args.algorithm == "ddqn":
        # define parameter grid to try out on algorithm
        params__1 = {'epsilon': 1.0, 'gamma': .99, 'learning_rate': 0.001,
                     'memory': 2000}
        params__2 = {'epsilon': 0.5, 'gamma': .99, 'learning_rate': 0.001,
                     'memory': 2000}
        params__3 = {'epsilon': 1.0, 'gamma': .66, 'learning_rate': 0.001,
                     'memory': 2000}
        params__4 = {'epsilon': 1.0, 'gamma': .99, 'learning_rate': 0.001,
                     'memory': 1000}

        param_list = [params__3, params__4]

        # execute on each parameter combination
        for params in param_list:
            args.model_save_path, args.result_save_path, args.video_save_path = get_paths(root=args.save_path,
                                                                                          algo=args.algorithm,
                                                                                          params=params)

            # add recording wrapper
            env = add_recording(env=env, save_path=args.video_save_path)
            loss, mean_over_last_100 = ddqn.train_ddqn(args, env=env, params=params)
            params_list.append(params)
            mean_over_last_100_list.append(mean_over_last_100)

    elif args.algorithm == "a2c":

        params__1 = {'epsilon': 1.0, 'gamma': .99, 'learning_rate': 0.0003, 'memory': 1000000}
        params__2 = {'epsilon': 1.0, 'gamma': .99, 'learning_rate': 0.001, 'memory': 1000000}
        params__3 = {'epsilon': 1.0, 'gamma': .99, 'learning_rate': 0.0001, 'memory': 1000000}

        param_list = [params__1, params__2, params__3]

        # execute on each parameter combination
        for params in param_list:
            args.model_save_path, args.result_save_path, args.video_save_path = get_paths(root=args.save_path,
                                                                                          algo=args.algorithm,
                                                                                          params=params)

            # add recording wrapper
            env = add_recording(env=env, save_path=args.video_save_path)
            loss, mean_over_last_100 = a2c.train_a2c(args, env=env, params=params)
            params_list.append(params)
            mean_over_last_100_list.append(mean_over_last_100)

    else:
        print("No such algorithm.")
        exit(-1)

    # Get best algorithm and best mean over 100 runs
    max_index = np.argmax(mean_over_last_100_list)
    best_file = open(args.save_path + "/" + args.algorithm + "/best_algo.txt", "a+")
    best_file.write(f"Params: {param_list[max_index]}, Score: {mean_over_last_100_list[max_index]} \n")
    best_file.flush()
    best_file.close()

    # Visualize
    # plt.plot([i + 1 for i in range(0, len(loss), 2)], loss[::2])
    # plt.show()


if __name__ == "__main__":
    main()
