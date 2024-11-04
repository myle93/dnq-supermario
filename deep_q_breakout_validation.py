import random
import numpy
import torch
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

from torch import Tensor
from torch.nn import Module, Conv2d, Linear, functional


class Network(Module):
    def __init__(self, inFrames: int, outputDimension: int):
        super().__init__()  # type: ignore
        # initializes layers with kaiming uniform
        self.conv2d_1 = Conv2d(
            in_channels=inFrames, out_channels=16, kernel_size=8, stride=4
        )
        self.conv2d_2 = Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.linear_1 = Linear(2592, 256)
        self.linear_2 = Linear(256, outputDimension)

    def forward(self, input: int):
        # input (1,84,84)
        x = self.conv2d_1(input)
        x = functional.relu(x)
        x = self.conv2d_2(x)
        x = functional.relu(x)
        x = x.flatten()
        x = self.linear_1(x)
        x = functional.relu(x)
        x = self.linear_2(x)
        # output (outputDimension)
        return x

    def forward_batch(self, input: Tensor):
        # input (Batch,1,84,84)
        x = self.conv2d_1(input)
        x = functional.relu(x)
        x = self.conv2d_2(x)
        x = functional.relu(x)
        x = x.flatten(start_dim=1)
        x = self.linear_1(x)
        x = functional.relu(x)
        x = self.linear_2(x)
        # output (Batch, outputDimension)
        return x

def pickAction(state):
    global Epsilon

    if(random.random()<Epsilon):
        return env.action_space.sample()
    else:
        return torch.argmax(BreakoutPlayer(state)).item()
    
def FrameskipStep(env, action, SkipFrames):
    fullReward = 0
    for _ in range(SkipFrames):
        observation, reward, terminated, truncated, info = env.step(action)
        fullReward += reward
        if(terminated or truncated):
            return observation, fullReward, terminated, truncated, info
    return observation, fullReward, terminated, truncated, info

Epsilon = 0.05

inFrames = 4
possibleActions = 4
# the agent only sees every nth frame (1, 1+n, 1+2n, etc.) so the agent doesnt see the n-1 frames in between
SkipFrames = 4

BreakoutPlayer = Network(inFrames,possibleActions)
BreakoutPlayer.load_state_dict(torch.load(f"./checkpoints/breakout_agent.pt", weights_only=True))

# validation enviroment

Epsilon = 0.05

env = gym.make("ALE/Breakout-v5", render_mode = "human")

env = gym.wrappers.ResizeObservation(env, (100,84))

env = gym.wrappers.GrayScaleObservation(env, keep_dim = True)

env = gym.wrappers.FrameStack(env, 4)

validationEpisodeReward = 0

with torch.no_grad():
    observation, info = env.reset()
    nextState = torch.tensor(numpy.array(observation)[:,8:8+84,:,:], dtype=torch.float32)
    nextState= nextState.reshape((1,4,84,84))

    terminated, truncated = False, False

    while not terminated and not truncated:
        state = nextState
        action = pickAction(state)
        observation, reward, terminated, truncated, info = FrameskipStep(env, action, SkipFrames)
        validationEpisodeReward += reward
        nextState = torch.tensor(numpy.array(observation)[:,8:8+84,:,:], dtype=torch.float32)
        nextState = nextState.reshape((1,4,84,84))

print(validationEpisodeReward)

env.close()