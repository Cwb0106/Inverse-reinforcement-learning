import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, maxaction=1):
        super(Actor, self).__init__()

        self.FC1 = nn.Linear(state_dim, hidden_size)
        self.FC2 = nn.Linear(hidden_size, hidden_size)
        self.FC3 = nn.Linear(hidden_size, action_dim)
        self.maxaction = maxaction

    def forward(self, state):
        x = F.relu(self.FC1(state))
        x = F.relu(self.FC2(x))
        action = torch.tanh(self.FC3(x)) * self.maxaction
        return action


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Q_Critic, self).__init__()

        self.FC1 = nn.Linear(state_dim+action_dim, hidden_size)
        self.FC2 = nn.Linear(hidden_size, hidden_size)
        self.FC3 = nn.Linear(hidden_size, 1)

        self.FC4 = nn.Linear(state_dim+action_dim, hidden_size)
        self.FC5 = nn.Linear(hidden_size, hidden_size)
        self.FC6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        Q1 = F.relu(self.FC1(state_action))
        Q1 = F.relu(self.FC2(Q1))
        Q1 = self.FC3(Q1)

        Q2 = F.relu(self.FC4(state_action))
        Q2 = F.relu(self.FC5(Q2))
        Q2 = self.FC6(Q2)
        return Q1, Q2

    def Q1(self, state, action):
        state_action = torch.cat([state, action], 1)

        Q1 = F.relu(self.FC1(state_action))
        Q1 = F.relu(self.FC2(Q1))
        Q1 = self.FC3(Q1)
        return Q1


class discriminator(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(discriminator, self).__init__()

        self.FC1 = nn.Linear(state_dim, hidden_size)
        self.FC2 = nn.Linear(hidden_size, hidden_size)
        self.FC3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.FC1(state))
        x = F.relu(self.FC2(x))
        out = torch.sigmoid(self.FC3(x))
        return out

