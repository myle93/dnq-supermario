#!/usr/bin/env python
# coding: utf-8

# In[11]:


import random
import numpy
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from collections import deque
import ale_py
from tqdm import tqdm
from pathlib import Path

current_path = Path(__file__).resolve().parent

gym.register_envs(ale_py)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


# In[12]:


class ReplayBuffer:
    def __init__(self, maxLen):
        self.Buffer = deque([], maxlen=maxLen)

    def add(self,data):
        self.Buffer.append(data)
        return
    
    def sample(self, batchSize):
        # returns a list with the batches as a list at each index
        x = random.sample(self.Buffer, batchSize)
        return list(zip(*x))
    
    def altSample(self, batchSize):
        # might work, might not
        indices = random.sample(range(len(self.Buffer)), batchSize)
        batch = [[self.Buffer[x][y] for x in indices] for y in range(len(self.Buffer[0]))]
        return batch


# In[13]:


class Network(torch.nn.Module):
    def __init__(self, inFrames, outputDimension):
        super().__init__()
        # initializes layers with kaiming uniform
        self.Conv2d_1 = torch.nn.Conv2d(in_channels=inFrames, out_channels=16, kernel_size=8, stride=4)
        self.Conv2d_2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)  
        self.Linear_1 = torch.nn.Linear(2592, 256)
        self.Linear_2 = torch.nn.Linear(256,outputDimension)
    def forward(self, input):
        # input (1,84,84)
        x = self.Conv2d_1(input)
        x = torch.nn.functional.relu(x)
        x = self.Conv2d_2(x)
        x = torch.nn.functional.relu(x)
        x = x.flatten()
        x = self.Linear_1(x)
        x = torch.nn.functional.relu(x)
        x = self.Linear_2(x)
        # output (outputDimension)
        return x
    def forwardBatch(self, input):
        # input (Batch,1,84,84)
        x = self.Conv2d_1(input)
        x = torch.nn.functional.relu(x)
        x = self.Conv2d_2(x)
        x = torch.nn.functional.relu(x)
        x = x.flatten(start_dim=1)
        x = self.Linear_1(x)
        x = torch.nn.functional.relu(x)
        x = self.Linear_2(x)
        # output (Batch, outputDimension)
        return x


# In[14]:


def pickAction(state):
    global Epsilon

    if(Epsilon > EpsilonEnd):
        Epsilon = Epsilon - ((EpsilonStart-EpsilonEnd)/EpsilonStepsbetweenStartandEnd)
    if(random.random()<Epsilon):
        return env.action_space.sample()
    else:
        return torch.argmax(BreakoutPlayer(state)).item()


# In[15]:


def FrameskipStep(env, action, SkipFrames):
    fullReward = 0
    for _ in range(SkipFrames):
        observation, reward, terminated, truncated, info = env.step(action)
        fullReward += reward
        if(terminated or truncated):
            return observation, fullReward, terminated, truncated, info
    return observation, fullReward, terminated, truncated, info


# In[16]:


def Visualization(StateBatch, actionBatch, rewardBatch, NextStateBatch, predictions, nextStateprediction, target):
    # input / output vizualization
    # state
    fig, ax = plt.subplots(1, 4, figsize=(13, 10))
    for i in range(inFrames):
        ax[i].imshow(StateBatch[0][i].cpu(), cmap='gray', vmin=0, vmax=255)
        plt.tight_layout()
    plt.show()
    # predictions for the moves
    print("Predictions for the moves: \nNOOP, FIRE, RIGHT, LEFT")
    print(predictions.cpu()[0])
    # actual move
    print("Actual move (might be different from highest predicted value because decision was made by old model or randomly)")
    print(actionBatch[0].cpu())
    # nextState
    fig, ax = plt.subplots(1, 4, figsize=(13, 10))
    for i in range(inFrames):
        plt.imshow(NextStateBatch[0][i].cpu(), cmap='gray', vmin=0, vmax=255)
        ax[i].imshow(NextStateBatch[0][i].cpu(), cmap='gray', vmin=0, vmax=255)
        plt.tight_layout()
    plt.show()
    # nextStatePredictions
    print("Prediction for the moves in the next State: \n: NOOP, FIRE, RIGHT, LEFT")
    print(nextStateprediction[0].cpu())
    # actual reward
    print("Actual reward")
    print(rewardBatch[0])
    # target
    print("Target (Prediction of actual move + gamma*Highest predicted action value in NextState)")
    print(target[0])


# In[17]:


def Optimizer(BreakoutPlayer, BreakoutBuffer, BatchSize, optimizer, lossFunction):
    # TD Learning
    if (len(BreakoutBuffer.Buffer)<BatchSize):
        return
    Batches = BreakoutBuffer.sample(BatchSize)
    StateBatch = torch.cat(Batches[0])
    actionBatch = torch.cat(Batches[1])
    rewardBatch = torch.cat(Batches[2])
    NextStateBatch = torch.cat(Batches[3])

    # zeroes the gradients because default behaviour in PT is to accumulate them
    for param in BreakoutPlayer.parameters():
        param.grad = None

    predictions = BreakoutPlayer.forwardBatch(StateBatch)
    prediction = predictions.gather(1, actionBatch)
    nextStateprediction = BreakoutPlayer.forwardBatch(NextStateBatch)
    nextStatepredictionMax = nextStateprediction.max(1).values.unsqueeze(1)
    target = rewardBatch + gamma*nextStatepredictionMax

    # in progress
    # if(i%???):
    #     Visualization(StateBatch, actionBatch, rewardBatch, NextStateBatch, predictions, nextStateprediction, target)

    loss = lossFunction(prediction,target)
    loss.backward()
    optimizer.step()

    return 


# In[18]:


EpsilonStart = 1
Epsilon = EpsilonStart
EpsilonStepsbetweenStartandEnd = 1000000
EpsilonEnd = 0.1

gamma = 0.9
BatchSize = 32

inFrames = 4
possibleActions = 4
# the agent only sees every nth frame (1, 1+n, 1+2n, etc.) so the agent doesnt see the n-1 frames in between
SkipFrames = 4

BufferLength = 1000000

agent = "Breakout_DQN_0"

BreakoutPlayer = Network(inFrames,possibleActions)
model_path = current_path / "models" / agent
if model_path.exists():
    BreakoutPlayer.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

BreakoutBuffer = ReplayBuffer(BufferLength)

optimizer = torch.optim.Adam(BreakoutPlayer.parameters(), lr=1e-4)

# what reduction to use? if you sum the gradients are bigger (also depend on batchSize then)
# just use mean for now
lossFunction = torch.nn.HuberLoss(reduction="mean")


# In[19]:


# sampling trajectories loop
if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5")

    env = gym.wrappers.ResizeObservation(env, (100,84))

    env = gym.wrappers.GrayScaleObservation(env, keep_dim = True)

    env = gym.wrappers.FrameStack(env, 4)

    observation, info = env.reset()
    nextState = torch.tensor(numpy.array(observation)[:,8:8+84,:,:], dtype=torch.float32)
    nextState= nextState.reshape((1,4,84,84))

    trainingSteps = 600000


    episodeReward = 0

    for i in tqdm(range(trainingSteps)):
        state = nextState
        action = pickAction(state)
        observation, reward, terminated, truncated, info = FrameskipStep(env, action, SkipFrames)
        episodeReward += reward
        nextState = torch.tensor(numpy.array(observation)[:,8:8+84,:,:], dtype=torch.float32)
        nextState = nextState.reshape((1,4,84,84))
        BreakoutBuffer.add([state,torch.tensor([[action]]),torch.tensor([[reward]]),nextState])

        Optimizer(BreakoutPlayer, BreakoutBuffer, BatchSize, optimizer, lossFunction)

        if terminated or truncated:
            observation, info = env.reset() 
            nextState = torch.tensor(numpy.array(observation)[:,8:8+84,:,:], dtype=torch.float32)
            nextState = nextState.reshape((1,4,84,84))
            episodeReward = 0
            print(i)
        if (i%10000 == 0):
            torch.save(BreakoutPlayer.state_dict(), current_path / "models" / agent)    

    env.close()

    # save model 
    torch.save(BreakoutPlayer.state_dict(), current_path / "models" / agent)

