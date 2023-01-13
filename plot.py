import argparse
import os
from utils import read_scores, plot_scores

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="Plotting of LunarLanding")

parser.add_argument(
    "--save_path", type=str, default="experiments", help="Path for model and logs"
)
parser.add_argument(
    "--algorithm", type=str, default="dqn", help="Algorithm"
)
args = parser.parse_args()


def main() -> None:
    directories = [name for name in os.listdir(args.save_path + "/" + args.algorithm)]
    for directory in directories:
        scores_path = args.save_path + "/" + args.algorithm + "/" + directory + "/result" + "/scores.txt"
        scores = read_scores(path=scores_path)
        plot_path = args.save_path + "/" + args.algorithm + "/" + directory + "/plot"
        os.makedirs(plot_path, exist_ok=True)
        plot_scores(_scores=scores, algo_name=args.algorithm, params=directory, save_path=plot_path + "/scores.png")


if __name__ == '__main__':
    main()
