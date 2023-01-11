import gym
import argparse
import warnings
warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="LunaLanding")

parser.add_argument("--continuous", type=bool, default=False, help="Continous Environment")

args = parser.parse_args()

## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():

    env = gym.make(
        "LunarLander-v2",
        continuous= args.continuous,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5,
        render_mode="human"
    )

    env.action_space.seed(42)

    observation, info = env.reset(seed=42)

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()