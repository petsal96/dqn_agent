import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import agent_code.dqn_agent3.features
import agent_code.dqn_agent3.rewards
import os.path
from collections import deque

GAMMA = 0.99
BATCH_SIZE = 32

MIN_MEMORY_BUFFER = 300
MAX_MEMORY_BUFFER = 10000

MIN_EPS = 0.1
EPS_STEPS = 500000

CRITERION = nn.SmoothL1Loss()
FEATURE = agent_code.dqn_agent3.features.calc_features
REWARD = agent_code.dqn_agent3.rewards.rewards

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvDQN(nn.Module):
    
    def __init__(self, num_inputs=6, num_actions=6):
        super(ConvDQN, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.fc3 = nn.Linear(2304, 512)
        self.fc4 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        return self.fc4(x)

    def copy_weights(self, conv_dqn):
        self.load_state_dict(conv_dqn.state_dict())

class DQNAgent:

    def __init__(self, train=False):
        self.Train = train
        self.MemoryBuffer = []
        self.Epsilon = 1.0

        self.Model = ConvDQN()
        self.Model.to(DEVICE)
        self.TargetModel = ConvDQN()
        self.TargetModel.to(DEVICE)
        self.Optimizer = torch.optim.RMSprop(self.Model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)    

    def CalculateFeatures(self, game_state):
        state = FEATURE(game_state)
        return state.to(DEVICE)
        
    def CalculateRewards(self, events):
        return REWARD(events)

    def PredictAction(self, x1):
        if random.random() < self.Epsilon and self.Train:
            return random.choice(ACTIONS)
        self.Model.eval()
        pred = self.Model(x1[None, :, :, :])
        action = ACTIONS[int(torch.argmax(pred))]
        return action

    def UpdateExplorationProbability(self):
        if len(self.MemoryBuffer) < MIN_MEMORY_BUFFER:
            return
        self.Epsilon -= (1.0-MIN_EPS)/EPS_STEPS
        if self.Epsilon < MIN_EPS:
            self.Epsilon = MIN_EPS
    
    def StoreExperience(self, old_game_state, self_action, new_game_state, reward, done):
        old_game_state = self.CalculateFeatures(old_game_state)
        new_game_state = self.CalculateFeatures(new_game_state)
        self.MemoryBuffer.append({
            "old_game_state": old_game_state,
            "self_action": self_action,
            "new_game_state": new_game_state,
            "reward": reward,
            "done": done
        })
        if len(self.MemoryBuffer) > MAX_MEMORY_BUFFER:
            self.MemoryBuffer.pop(0)

    def UpdateTargetModel(self):
        self.TargetModel.copy_weights(self.Model)
    
    def GetBatch(self, batch_size):
        batch = random.sample(self.MemoryBuffer, batch_size)

        old_game_states = []
        self_actions = []
        new_game_states = []
        rewards = []
        dones = []
        for exp in batch:
            old_game_states.append(exp["old_game_state"])
            self_actions.append(ACTIONS.index(exp["self_action"]))
            new_game_states.append(exp["new_game_state"])
            rewards.append(exp["reward"])
            dones.append(int(exp["done"]))

        old_game_states = torch.stack(old_game_states).to(DEVICE)
        self_actions = torch.LongTensor(self_actions).unsqueeze(1).to(DEVICE)
        new_game_states = torch.stack(new_game_states).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        dones = torch.IntTensor(dones).unsqueeze(1).to(DEVICE)

        return old_game_states, self_actions, new_game_states, rewards, dones

    def UpdateWeights(self):
        if len(self.MemoryBuffer) < MIN_MEMORY_BUFFER:
            return

        old_game_states, self_actions, new_game_states, rewards, dones = self.GetBatch(BATCH_SIZE)

        self.Model.train()
        self.TargetModel.train()
        
        q_current = self.Model(old_game_states).gather(1, self_actions)
       
        q_target = self.TargetModel(new_game_states).detach().max(1)[0]
        q_target = q_target.unsqueeze(1)
        q_target = rewards + (1-dones)*GAMMA*q_target
        loss = CRITERION(q_current, q_target)

        self.Optimizer.zero_grad()
        loss.backward()
        for param in self.Model.parameters():
            param.grad.data.clamp_(-1,1)
        self.Optimizer.step()

    def Save(self):        
        torch.save(self.Model.state_dict(), "model/DQNTorch.obj")

    def Load(self):
        if os.path.exists("model/DQNTorch.obj"):
            self.Model.load_state_dict(torch.load("model/DQNTorch.obj"))
