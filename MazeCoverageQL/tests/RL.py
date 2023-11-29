import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.distributions.categorical import Categorical
import numpy as np
import random
from collections import deque


class Actor:
    def __init__(self, learning_rate, hidden_d, n_actions, share_backbone=False):
        self.learning_rate = learning_rate
        # Initialize actor network with random weights
        self.network = nn.Sequential(
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate)
        
    
    def update_weights(self, state, action, advantage):
        # Convert the state to a PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Get the action probabilities from the actor network
        action_probs = self.network(state_tensor)

        # Compute the log-likelihood of the selected action
        log_prob = torch.log(action_probs[action])

        # Compute the loss using the advantage
        loss = -log_prob * advantage

        # Perform gradient descent on the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class Critic:
    def __init__(self, learning_rate, hidden_d, n_actions, share_backbone=False):
        self.learning_rate = learning_rate
        # Initialize actor network with random weights
        self.network = nn.Sequential(
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate)
        
    def update_weights(self, state, action, advantage):
        # Convert the state to a PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Get the action probabilities from the actor network
        action_probs = self.network(state_tensor)

        # Compute the log-likelihood of the selected action
        log_prob = torch.log(action_probs[action])

        # Compute the loss using the advantage
        loss = -log_prob * advantage

        # Perform gradient descent on the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




class Agent:
    def __init__(self, learning_rate_actor, learning_rate_critic, buffer_size, hidden_d, n_actions):
        self.actor = Actor(learning_rate_actor, hidden_d, n_actions)
        self.critic = Critic(learning_rate_critic, hidden_d, n_actions)
        self.buffer = deque(maxlen=buffer_size)
        # You can also define other hyperparameters and components here

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_from_buffer(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def train(self, environment, num_episodes, batch_size, discount_factor):
        for episode in range(num_episodes):
            state = environment.reset()
            done = False

            while not done:
                action = self.actor.choose_action(state)
                next_state, reward, done, _ = environment.step(action)

                # Store the experience in the replay buffer
                self.remember(state, action, reward, next_state, done)

                # Sample a batch from the replay buffer
                states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)

                # Update the critic using the batch
                deltas = [r + discount_factor * self.critic.predict(next_s) - self.critic.predict(s) for r, s, next_s in zip(rewards, states, next_states)]
                self.critic.update_weights(states, deltas)

                # Compute advantages for the actor update
                advantages = deltas

                # Update the actor using the batch
                self.actor.update_weights(states, actions, advantages)

                state = next_state

    def act(self, state):
        return self.actor.choose_action(state)

