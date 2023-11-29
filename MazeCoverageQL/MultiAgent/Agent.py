from DQN import Qnet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  

    def add(self, state, action, reward, next_state, done):  
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  
        return len(self.buffer)


class Agent:
    def __init__(self,number_of_steps,LearningRate,n_actions,hidden_channels,state_dim,gamma,epsilon,target_update,device):
        self.position = None
        self.reward = 0
        self.number_of_steps = number_of_steps
        self.q_net = Qnet(LearningRate,n_actions,hidden_channels,state_dim).to(device)
        self.target_q_net = Qnet(LearningRate,n_actions,hidden_channels,state_dim).to(device)
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.target_update = target_update  
        self.count = 0  
        self.device = device
        self.position_grid = None
        self.action_dim = n_actions
        self.other_agent_positions = None
        self.total_steps = 0
        self.buffer = ReplayBuffer(10000)
        self.coverage_grid = None

    def is_alive(self):
        return self.total_steps < self.number_of_steps

    def reset(self,environment):
        self.coverage_grid = np.zeros(environment.grid_size)
        self.position = (np.random.randint(0, environment.grid_size[0]), np.random.randint(0, environment.grid_size[1]))
        self.position_grid = np.zeros(environment.grid_size)
        self.position_grid[self.position] = 1
        self.reward = 0
        self.total_steps = 0

    def get_other_agents(self,environment):
        self.other_agent_positions = environment.get_other_agents(self.position)

    def take_action(self, state): 
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = state.reshape((1, 4, state.shape[0], state.shape[1]))
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action


    def update_epsilon(self,decay_rate):
        self.epsilon = max(0.0000001, self.epsilon - decay_rate)

    def update_lr(self,decay_rate):
        self.q_net.learning_rate = max(0.01, self.q_net.learning_rate - decay_rate)


    def update(self, transition_dict):
        states = np.stack(transition_dict['states'])
        states = states.reshape((states.shape[0], 4, states.shape[1], states.shape[2]))
        states = torch.tensor(states,
                              dtype=torch.float).to(self.device)
        
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = np.stack(transition_dict['states'])

        next_states = next_states.reshape((next_states.shape[0], 4, next_states.shape[1], next_states.shape[2]))
        next_states = torch.tensor(next_states,
                                   dtype=torch.float).to(self.device)
        
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  
        self.q_net.optimizer.zero_grad()  
        dqn_loss.backward()  
        self.q_net.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  
        self.count += 1

