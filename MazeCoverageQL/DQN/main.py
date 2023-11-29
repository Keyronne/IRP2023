import utils
from RL2 import Actor, Critic, Agent
from Environment import GridWorld
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = utils.get_device()
print("device: ", device)
buffer_size=1000
grid_world = GridWorld(grid_size=(5, 5), obstacle_density=0.2)
agent = Agent(learning_rate_actor=0.0001, learning_rate_critic=0.0001, buffer_size=1000, hidden_d=5, n_actions=4, hidden_channels=5, device=device)

num_episodes = 1000
batch_size = 32
discount_factor = 0.99


def train_off_policy_agent(env, num_episodes, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.actor.choose_action(state,device)
                    next_state, reward, done, _ = env.step(action)
                    reward -= 0.01
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if agent.buffer_size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = agent.sample_from_buffer(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list



return_list = train_off_policy_agent(grid_world, num_episodes, 100, batch_size)