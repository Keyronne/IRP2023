import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
import random
from collections import deque
import numpy as np
import sys
from torchvision import models


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

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x



class Actor(nn.Module):
    def __init__(self, learning_rate, n_actions, hidden_channels):
        super(Actor, self).__init__()
        self.learning_rate = learning_rate
        self.input_channels = 1  
        self.hidden_channels = hidden_channels
        self.n_actions = n_actions
        self.network = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_channels * 5 * 5  , self.n_actions),
            #PrintLayer(),
            nn.Softmax(dim=1)
        )
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        return self.network(state)

    def choose_action(self, state, device):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
        action_probs = self.network(state)
        try:
            return torch.multinomial(action_probs, 1).item()
        except RuntimeError:
            print(state)
            sys.exit(0)


class Critic(nn.Module):
    def __init__(self, learning_rate, hidden_channels):
        super(Critic, self).__init__()
        self.learning_rate = learning_rate
        self.input_channels = 1  
        self.hidden_channels = hidden_channels
        self.network = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_channels * 5 * 5  , 1),
        )
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        return self.network(state)

    def predict(self, state):
        return self.network(state)


class A2CAgent:
    def __init__(self, learning_rate_actor, learning_rate_critic, buffer_size, hidden_channels, n_actions, device):
        self.actor = Actor(learning_rate_actor, n_actions, hidden_channels ).to(device)
        self.critic = Critic(learning_rate_critic, hidden_channels).to(device)
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.gamma = 0.98

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_from_buffer(self, batch_size):
        if len(self.buffer) < batch_size:
            # If the buffer size is smaller than the batch size, return all experiences in the buffer
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def buffer_size(self):
        return len(self.buffer)



    def update(self, transition_dict):
        states = np.stack(transition_dict['states'])
        states = states.reshape((states.shape[0], 1, states.shape[1], states.shape[2]))
        states = torch.tensor(states,
                              dtype=torch.float).to(self.device)
        
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = np.stack(transition_dict['states'])

        next_states = next_states.reshape((next_states.shape[0], 1, next_states.shape[1], next_states.shape[2]))
        next_states = torch.tensor(next_states,
                                   dtype=torch.float).to(self.device)
        
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        actor_loss.backward()  
        critic_loss.backward()  
        self.actor.optimizer.step()  
        self.critic.optimizer.step()  

class Up(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )

        
        
    def forward(self, x1, x2):
        # Input size - Batch_Size X Channel X Height of Activation Map  X Width of Activation Map
        # Upsample using bilinear mode and scale it to twice its size
        x1 = self.upsample(x1)
        # in 4D array - matching the last two in case of 5D it will take 
        # last three dimensions
        difference_in_X = x1.size()[2] - x2.size()[2]
        difference_in_Y = x1.size()[3] - x2.size()[3]
        # Padding it with the required value
        x2 = F.pad(x2, (difference_in_X // 2, int(difference_in_X / 2),
                        difference_in_Y // 2, int(difference_in_Y / 2)))
        # concat on channel axis
        x = torch.cat([x2, x1], dim=1)
        # Use convolution
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Input size - Batch_Size X Channel X Height of Activation Map  X Width of Activation Map
        # Downsample First
        x = F.max_pool2d(x,2)
        # Use convolution
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, channel_in, classes):
        super(UNet, self).__init__()
        self.input_conv = self.conv = nn.Sequential(
            nn.Conv2d(channel_in, 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 32)
        self.up1 = Up(64, 16)
        self.up2 = Up(32, 8)
        self.up3 = Up(16, 4)
        self.output_conv = nn.Conv2d(4, classes, kernel_size = 1)
        
    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.output_conv(x)
        return F.sigmoid(output)
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.init.xavier_uniform(m.weight, gain=np.sqrt(2.0))
        nn.init.init.constant(m.bias, 0.1)

class Qnet(nn.Module):
    def __init__(self, learning_rate, n_actions, hidden_channels,state_dim):
        super(Qnet, self).__init__()
        self.learning_rate = learning_rate
        self.input_channels = 2 
        self.hidden_channels = hidden_channels
        self.layer2Size = 64
        self.n_actions = n_actions
        self.network = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.layer2Size, kernel_size=3, stride=1, padding=1),
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
            #PrintLayer(),
        )
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        return self.network(state)

class DQNAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device, type):
        self.action_dim = action_dim
        self.q_net = Qnet(learning_rate, self.action_dim, hidden_dim,state_dim).to(device)  
        
        self.target_q_net = Qnet(learning_rate,self.action_dim,hidden_dim,state_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.target_update = target_update  
        self.count = 0  
        self.device = device
        self.type = type

    def take_action(self, state): 
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = state.reshape((1, 2, state.shape[0], state.shape[1]))
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update_epsilon(self,decay_rate):
        self.epsilon = max(0.0000001, self.epsilon - decay_rate)

    def update_lr(self,decay_rate):
        self.q_net.learning_rate = max(0.01, self.q_net.learning_rate - decay_rate)


    def update(self, transition_dict):
        states = np.stack(transition_dict['states'])
        states = states.reshape((states.shape[0], 2, states.shape[1], states.shape[2]))
        states = torch.tensor(states,
                              dtype=torch.float).to(self.device)
        
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = np.stack(transition_dict['states'])

        next_states = next_states.reshape((next_states.shape[0], 2, next_states.shape[1], next_states.shape[2]))
        next_states = torch.tensor(next_states,
                                   dtype=torch.float).to(self.device)
        
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)  

        if self.type == "DoubleDQN":
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1) 
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)

        else:
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

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_channels, n_actions):
        super(PolicyNet, self).__init__()
        self.input_channels = 2 
        self.hidden_channels = hidden_channels
        self.layer2Size = 64
        self.n_actions = n_actions
        self.network = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.layer2Size, kernel_size=3, stride=1, padding=1),
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
            #PrintLayer(),
        )


    def forward(self, state):
        return self.network(state)

class ValueNet(torch.nn.Module):
    def __init__(self,state_dim, hidden_channels):
        super(ValueNet, self).__init__()

        self.input_channels = 2 
        self.hidden_channels = hidden_channels
        self.layer2Size = 64
        self.network = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.layer2Size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.layer2Size * state_dim , 256),
            nn.ReLU(),
            nn.Linear(256 , 128),
            nn.ReLU(),
            nn.Linear(128 , 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #PrintLayer(),
        )


    def forward(self, state):
        return self.network(state)
    
class PPO:

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  
        self.eps = eps  
        self.device = device

    def take_action(self, state):
        state = state.reshape((1, 2, state.shape[0], state.shape[1]))
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def compute_advantage(self,gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta[0]
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def update(self, transition_dict):
        states = np.stack(transition_dict['states'])
        states = states.reshape((states.shape[0], 2, states.shape[1], states.shape[2]))
        states = torch.tensor(states,
                              dtype=torch.float).to(self.device)
        
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = np.stack(transition_dict['states'])

        next_states = next_states.reshape((next_states.shape[0], 2, next_states.shape[1], next_states.shape[2]))
        next_states = torch.tensor(next_states,
                                   dtype=torch.float).to(self.device)
        
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda,td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  
            actor_loss = torch.mean(-torch.min(surr1, surr2))  
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()



    