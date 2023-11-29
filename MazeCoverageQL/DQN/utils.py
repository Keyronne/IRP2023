import torch
import numpy as np


def get_device():
    """Gets the device (GPU if any) and logs the type"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

colors = {-1:(0,0,0), 0:(255,255,255), 1:(0,255,0), 2:(0,0,255), 3:(255,0,0)}
def draw_gridworld(state, grid_size):
    img = np.zeros((grid_size[0], grid_size[1], 3), dtype=np.uint8)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            img[i,j,:] = colors[state[i,j,1]]
    return img