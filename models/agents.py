from pathlib import Path
from models.base_model import Network
from torch import cat, no_grad
from numpy import array
from random import random
from typing import Any, Union
from torch import Tensor, argmax, device, float32, load, save, tensor  # type: ignore
from torch.optim import Adam
from torch.nn import HuberLoss
from torch.cuda import is_available
from gymnasium.wrappers.frame_stack import FrameStack, LazyFrames
from models.buffer import ReplayBuffer
from tqdm.auto import tqdm
from pydantic import BaseModel


class BreakOutConfig(BaseModel):

    class Env(BaseModel):
        game: str
        n_stack: int
        keep_dim: bool
        obs_shape: tuple[int, int]

    class Optimizer(BaseModel):
        validation_epsilon: float
        epsilon_start: float
        epsilon_end: float
        epsilon_steps_between_start_and_end: int
        gamma: float
        """Momentum of the optimizer"""
        lr: float

    class Train(BaseModel):
        batch_size: int
        buffer_length: int
        in_frames: int
        possible_actions: int
        steps: int
        frame_skip: int
        validate_while_training: bool = True
        """If true, the agent will validate while training and save the model if the validation reward is better than the previous best reward"""
        checkpoint_steps: int = 1000
        """The number of steps between each checkpoint save and each validation run, if validate_while_training is true"""

    agent_name: str
    env: Env
    optimizer: Optimizer
    train: Train


class BreakOutAgent:

    def __init__(
        self,
        config: BreakOutConfig,
        env: FrameStack,
        save_path: Path = Path(__file__).resolve().parent.parent
        / "checkpoints"
        / "breakout.pkl",
    ):
        self.device = "cuda" if is_available() else "cpu"
        self.save_path = save_path
        self.model = Network(config.train.in_frames, config.train.possible_actions).to(
            self.device
        )
        if save_path.exists():
            self.load_model()
        self.config = config
        self.train_epsilon: float = config.optimizer.epsilon_start
        self.validation_epsilon: float = config.optimizer.validation_epsilon
        self.env = env
        self.buffer = ReplayBuffer(config.train.buffer_length)
        self.optimizer = Adam(self.model.parameters(), config.optimizer.lr)
        self.loss_function = HuberLoss(reduction="mean")

    def _update_epsilon(self):
        if self.train_epsilon > self.config.optimizer.epsilon_end:
            self.train_epsilon = self.train_epsilon - (
                (
                    self.config.optimizer.epsilon_start
                    - self.config.optimizer.epsilon_end
                )
                / self.config.optimizer.epsilon_steps_between_start_and_end
            )

    def _pick_action(self, state: Tensor, epsilon: float) -> Union[int, float, bool]:
        if random() < epsilon:
            return self.env.action_space.sample()  # type: ignore
        else:
            return argmax(self.model(state)).item()

    def _frame_skip_step(
        self, action: Union[int, float, bool]
    ) -> tuple[LazyFrames | None, float, bool, bool, dict[str, Any]]:
        fullReward = 0
        observation: LazyFrames | None = None
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        for _ in range(self.config.train.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)  # type: ignore
            fullReward += float(reward)
            if terminated or truncated:
                return observation, fullReward, terminated, truncated, info
        return observation, fullReward, terminated, truncated, info

    def save_model(self):
        save(self.model.state_dict(), self.save_path)

    def load_model(self):
        self.model.load_state_dict(
            load(self.save_path, map_location=device(self.device))
        )

    def optimize(
        self,
    ):
        # TD Learning
        if len(self.buffer.buffer) < self.config.train.batch_size:
            return
        batch = self.buffer.sample(self.config.train.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = (
            cat(batch[0]),
            cat(batch[1]),
            cat(batch[2]),
            cat(batch[3]),
        )

        # zeroes the gradients because default behaviour in PT is to accumulate them
        for param in self.model.parameters():
            param.grad = None

        predictions = self.model.forward_batch(state_batch)
        prediction = predictions.gather(1, action_batch)
        nextStateprediction = self.model.forward_batch(next_state_batch)
        nextStatepredictionMax = nextStateprediction.max(1).values.unsqueeze(1)
        target = reward_batch + self.config.optimizer.gamma * nextStatepredictionMax

        # in progress
        # if(i%???):
        #     Visualization(StateBatch, actionBatch, rewardBatch, NextStateBatch, predictions, nextStateprediction, target)

        loss = self.loss_function(prediction, target)
        loss.backward()
        self.optimizer.step()  # type: ignore

        return

    def train(self):
        observation, _ = self.env.reset()  # type: ignore
        nextState = tensor(
            array(observation)[:, 8 : 8 + 84, :, :], dtype=float32
        ).reshape((1, 4, 84, 84))
        state = nextState
        episode_reward = 0
        validation_episode_reward = 0

        for i in tqdm(range(self.config.train.steps), position=0, leave=True):
            state = nextState
            self._update_epsilon()
            action = self._pick_action(state, self.train_epsilon)
            observation, reward, terminated, truncated, _ = self._frame_skip_step(
                action
            )
            episode_reward += reward
            nextState = tensor(array(observation)[:, 8 : 8 + 84, :, :], dtype=float32)
            nextState = nextState.reshape((1, 4, 84, 84))
            self.buffer.add([state, tensor([[action]]), tensor([[reward]]), nextState])

            self.optimize()

            if terminated or truncated:
                observation, _ = self.env.reset()  # type: ignore
                nextState = tensor(
                    array(observation)[:, 8 : 8 + 84, :, :], dtype=float32
                )
                nextState = nextState.reshape((1, 4, 84, 84))
                episode_reward = 0
            if (
                i % self.config.train.checkpoint_steps == 0
                or i == self.config.train.steps - 1
            ):
                if self.config.train.validate_while_training:
                    tqdm.write(
                        f"Validation at step: {i+1}, Train epsilon: {self.train_epsilon}"
                    )
                    current_validate_reward = self.validate()
                    if current_validate_reward > validation_episode_reward:
                        validation_episode_reward = current_validate_reward
                        self.save_model()
                else:
                    self.save_model()

    def validate(self) -> float:
        validation_episode_reward = 0

        with no_grad():
            observation, _ = self.env.reset()  # type: ignore
            nextState = tensor(
                array(observation)[:, 8 : 8 + 84, :, :], dtype=float32
            ).reshape((1, 4, 84, 84))
            state = nextState
            terminated, truncated = False, False
            step = 0
            while not terminated and not truncated:
                step += 1
                state = nextState
                self._update_epsilon()
                action = self._pick_action(state, self.validation_epsilon)
                observation, reward, terminated, truncated, _ = self._frame_skip_step(
                    action
                )
                validation_episode_reward += reward
                nextState = tensor(
                    array(observation)[:, 8 : 8 + 84, :, :], dtype=float32
                )
                nextState = nextState.reshape((1, 4, 84, 84))
            tqdm.write(f"Validation Reward: {validation_episode_reward}")
            return validation_episode_reward
