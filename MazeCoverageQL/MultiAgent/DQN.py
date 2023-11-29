import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
import random
from collections import deque
import numpy as np
import sys





class Qnet(nn.Module):
    def __init__(self, learning_rate, n_actions, hidden_channels,state_dim):
        super(Qnet, self).__init__()
        self.learning_rate = learning_rate
        self.input_channels = 4 
        self.hidden_channels = hidden_channels
        self.layer2Size = 64
        self.n_actions = n_actions
        self.network = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.layer2Size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.layer2Size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.layer2Size * state_dim , 256),
            nn.ReLU(),
            nn.Linear(256 , 128),
            nn.ReLU(),
            nn.Linear(128 , 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
            nn.Softmax(dim=1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        return self.network(state)

    