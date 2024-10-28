import argparse
import gymnasium
from gymnasium import make  # type: ignore
from gymnasium.wrappers import FrameStack, ResizeObservation, GrayScaleObservation  # type: ignore
import ale_py
from models.agents import BreakOutAgent, BreakOutConfig
import yaml

gymnasium.register_envs(ale_py)


def get_env_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Deep Q Breakout model")
    parser.add_argument("-c", "--configpath", type=str, help="Path to the config file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_env_args()
    config_path = args.configpath

    with open(config_path) as file:
        yaml_data = yaml.safe_load(file)
    config = BreakOutConfig.model_validate(yaml_data)

    env = make(config.env.game)  # type: ignore
    env = ResizeObservation(env, config.env.obs_shape)
    env = GrayScaleObservation(env, keep_dim=config.env.keep_dim)
    env = FrameStack(env, config.env.n_stack)

    agents = BreakOutAgent(config, env)
    agents.train()
    agents.validate()

    env.close()
