import argparse
import os

from utils import read_scores, calculate_mean

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="Plotting of LunarLanding")

parser.add_argument(
    "--save_path", type=str, default="experiments", help="Path for model and logs"
)
parser.add_argument(
    "--algorithm", type=str, default="a2c", help="Algorithm"
)
args = parser.parse_args()


def main():
    directories = [name for name in os.listdir(args.save_path + "/" + args.algorithm)]
    for directory in directories:
        scores_path = args.save_path + "/" + args.algorithm + "/" + directory + "/result" + "/scores.txt"
        scores = read_scores(path=scores_path)
        mean_path = args.save_path + "/" + args.algorithm + "/" + directory + "/result"
        os.makedirs(mean_path, exist_ok=True)
        calculate_mean(_scores=scores, mean_path=mean_path, params=directory)


if __name__ == '__main__':
    main()
