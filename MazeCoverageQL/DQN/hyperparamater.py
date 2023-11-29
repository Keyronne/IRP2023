import random
import numpy as np
import collections
from stqdm import stqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from RL2 import ReplayBuffer, DQNAgent
from Environment import GridWorld
from utils import get_device, moving_average, draw_gridworld
import streamlit as st
st.set_page_config(layout="wide")
device = get_device()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)




st.title('Hyperparameter Tuning')
lr = st.slider('learning rate', 0.0001, 0.01, 0.001, 0.0001)
num_episodes = st.slider('num_episodes', 100, 10000, 1000, 100) +1
hidden_dim = 128
gamma = st.slider('gamma', 0.9, 0.99, 0.9, 0.01)
epsilon = 1
target_update = st.slider('target_update', 10, 100, 10, 10)
buffer_size = 10000
minimal_size = 500
batch_size = 256
gridsize = st.slider('gridsize', 3, 10, 4, 1)
max_steps = gridsize * gridsize * 4
st.write('learning rate:', lr)
st.write('num_episodes:', num_episodes - 1)
st.write('gamma:', gamma)
st.write('target_update:', target_update)
st.write('gridsize:', gridsize)
st.write('max_steps:', max_steps)

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

st.title('Train')
if st.button('Start'):
    empty = st.empty()
    for i in range(10):
        with stqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
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
                    agent.update_epsilon(1/num_episodes)            
                return_list.append(episode_return)
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
                        '%.3f' % np.mean(steps_avg[-10:])
                    })
                pbar.update(1)


            empty.line_chart(return_list)



    mv_return = moving_average(return_list, 11)
    st.line_chart(mv_return)
st.cache_resource

st.title('Test')
startposition = st.slider('startposition', 0, gridsize * gridsize -1, 0, 1)
startposition = ((startposition//gridsize), (startposition%gridsize))
print(startposition)
if st.button('Test'):
    done = False
    env = GridWorld((gridsize,gridsize),0)


    state = env.reset_GivenPosition(startposition)
    states = []
    rewards = []
    rewards.append(0)
    states.append(state.copy())
    max_steps = gridsize * gridsize
    steps = 0
    element = st.empty()
    element2 = st.empty()
    while not done:
        action = agent.take_action(state)
        state, reward, done, _ = env.step(action,steps)
        states.append(state.copy())
        rewards.append(rewards[-1] + reward)
        steps += 1
        element.text('Steps: %d' % steps) 
        if steps > max_steps:
            element2.write('Failed')
            break

    
    fig, axs = plt.subplots(1, len(states), figsize=(15, 15))
    for i, state in enumerate(states):
        img = draw_gridworld(state, env.grid_size)
        axs[i].imshow(img)
        axs[i].set_title('State {}'.format(i))
        axs[i].axis('on')
        #draw grid
        for j in range(0, env.grid_size[0]):
            axs[i].axhline(y=j-0.5, color='k', linestyle='-')
            axs[i].axvline(x=j-0.5, color='k', linestyle='-')

    st.pyplot(fig)


    