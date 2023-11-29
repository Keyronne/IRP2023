import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from RL2 import ReplayBuffer, DQNAgent
from Environment import GridWorld
from utils import get_device, moving_average, draw_gridworld
import json
import os

device = get_device()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

learning_rate = np.arange(0.0002, 0.01, 0.0001)
num_episodes = 1000
hidden_dim = 128
Gamma = np.arange(0.9, 0.99, 0.001)
epsilon = 1
target_update = 100
buffer_size = 10000
minimal_size = 500
batch_size = 256
gridsize = 10
max_steps = gridsize * gridsize * 20

for gamma in Gamma:
    for lr in learning_rate:
        
        directory = f'output/{str(lr).replace(".","_")[:6]}_{str(gamma).replace(".","_")}'
        try:
            os.mkdir(directory)
        except Exception as e:
            print(e)
            pass

        env = GridWorld((gridsize,gridsize),0)
        replay_buffer = ReplayBuffer(buffer_size)
        state_dim = env.observation_space
        action_dim = env.action_space
        agent = DQNAgent(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                    target_update, device)
        done_counter = 0
        return_list = []
        steps_avg = []
        episode = 0

        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    episode += 1
                    episode_return = 0
                    state = env.reset()
                    done = False
                    steps = 0
                    while not done:
                        if steps > max_steps:
                            break
                        steps += 1

                        action = agent.take_action(state)
                        next_state, reward, done, _ = env.step(action, steps)
                        replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d
                            }
                            agent.update(transition_dict)
                        if done:
                            steps_avg.append(steps)
                            done_counter += 1
                    if replay_buffer.size() > minimal_size:
                        agent.update_epsilon(1.1/num_episodes)            
                    return_list.append(episode_return)
                    if len(steps_avg) > 10:
                        stepmean = np.mean(steps_avg[-10:])
                    else:
                        stepmean = 0
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return':
                            '%.3f' % np.mean(return_list[-10:]),
                            'Theoretical Max Return':
                            '%.3f' % env.theoretical_max,
                            'epsilon':
                            '%.3f' % agent.epsilon,
                            'times completed':
                            '%d' % done_counter,
                            'avg steps':
                            '%.3f' % stepmean,
                            'lr':
                            '%.4f' % lr,
                            'gamma':
                            '%.3f' % gamma

                        })
                    pbar.update(1)



        episodes_list = list(range(len(return_list)))
        fig, ax = plt.subplots()
        ax.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')

        mv_return = moving_average(return_list, 5)
        ax.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.savefig(f'{directory}/mvaverage.png')


        positiondict = {}
        env = GridWorld((gridsize,gridsize),0)
        stepdict = {}
        for i in range(gridsize):
            for j in range(gridsize):
                done = False
                state = env.reset_GivenPosition((i,j))
                states = []
                rewards = []
                rewards.append(0)
                states.append(state.copy().tolist())
                max_steps = gridsize * gridsize + 10
                steps = 0
                position = i*gridsize+j
                while not done:
                    action = agent.take_action(state)
                    state, reward, done, _ = env.step(action,steps)
                    states.append(state.copy().tolist())
                    rewards.append(rewards[-1] + reward)
                    steps += 1

                    if steps > max_steps:
                        done = True
                if steps <= max_steps:
                    
                    positiondict[position] = states

                stepdict[position] = steps

        with open(fr'{directory}/positiondict.json', 'w') as f:
            f.write(json.dumps(positiondict,ensure_ascii=True,indent=4))

        with open(fr'{directory}/stepdict.json', 'w') as f:
            f.write(json.dumps(stepdict,ensure_ascii=True,indent=4))


    