import numpy as np

class GridWorld:
    def __init__(self, grid_size=(5, 5), POI_density=0.2):
        self.grid_size = grid_size
        self.POI_density = POI_density
        self.state = None
        self.position_grid = None
        self.observation_space = self.grid_size[0] * self.grid_size[1]
        self.reward_grid = None
        self.reward_indices = self.get_POI()
        self.action_space = 4
        self.theoretical_max = None
        self.visited_grid = None

    def reset(self):
        self.state = np.zeros(self.grid_size)
        self.position_grid = np.zeros(self.grid_size)
        self.agent_position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        #self.agent_position = (0,0)
        self.place_POI()
        self.theoretical_max = self.theoretical_Maximum()
        self.position_grid[self.agent_position] = 1
        self.state[self.agent_position] = 1
        self.visited_grid = np.zeros((self.grid_size[0] + 1, self.grid_size[1] + 1))
        return np.stack((self.position_grid,self.state), axis= 2)
    
    def reset_GivenPosition(self,pos):
        self.state = np.zeros(self.grid_size)
        self.position_grid = np.zeros(self.grid_size)
        self.agent_position = pos
        self.place_POI()
        self.theoretical_max = self.theoretical_Maximum()
        self.position_grid[self.agent_position] = 1
        self.state[self.agent_position] = 1
        return np.stack((self.position_grid,self.state), axis= 2)


    def get_POI(self):
        self.reward_grid = np.zeros(self.grid_size)
        num_POI = int(self.POI_density * np.prod(self.grid_size))
        POI_indices = np.random.choice(np.prod(self.grid_size), num_POI, replace=False)
        POI_indices = np.unravel_index(POI_indices, self.grid_size)
        return POI_indices

    def place_POI(self):
        self.reward_grid = np.zeros(self.grid_size)
        self.reward_grid[self.reward_indices] = 2


    def step(self, action,step):

        agent_row, agent_col = self.agent_position
        if action == 0:  # Up
            next_row, next_col = agent_row - 1, agent_col
        elif action == 1:  # Down
            next_row, next_col = agent_row + 1, agent_col
        elif action == 2:  # Left
            next_row, next_col = agent_row, agent_col - 1
        elif action == 3:  # Right
            next_row, next_col = agent_row, agent_col + 1
        reward = -0.05
        self.visited_grid[next_row,next_col] += 1
        if 0 <= next_row < self.grid_size[0] and 0 <= next_col < self.grid_size[1]:
            # Move is valid

            self.position_grid[self.agent_position] = 0
            self.agent_position = (next_row, next_col)
            self.position_grid[self.agent_position] = 1


            if self.state[self.agent_position] != 1:
                reward += (self.reward_grid[self.agent_position] + 1) * 10
            if self.state[self.agent_position] == 1:
                reward -= 1 


            self.reward_grid[self.agent_position] = 0
        else:
            # Move is invalid
            reward -= 1 

        self.state[self.agent_position] = 1
            
        done = self.state.all() == 1
        if done:
            reward += 200 *  ((self.observation_space - 1) / step)
        #self.state[self.agent_position] = -1

        return np.stack((self.position_grid,self.state), axis= 2), reward, done, {}

    def theoretical_Maximum(self):
        reward = 0
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.reward_grid[i][j] == 2:
                    reward += 3
                elif self.reward_grid[i][j] == 0:
                    reward += 1
        reward += 200
        reward -= 0.005 * self.grid_size[0] * self.grid_size[1]
        return reward



    def reset_Qlearning(self):
        self.state = np.zeros(self.grid_size)
        self.agent_position = (0,0)
        self.place_POI()
        self.theoretical_max = self.theoretical_Maximum()
        return self.state
    

    def step_Qlearning(self, action):
        agent_row, agent_col = self.agent_position
        if action == 0:  # Up
            next_row, next_col = agent_row - 1, agent_col
        elif action == 1:  # Down
            next_row, next_col = agent_row + 1, agent_col
        elif action == 2:  # Left
            next_row, next_col = agent_row, agent_col - 1
        elif action == 3:  # Right
            next_row, next_col = agent_row, agent_col + 1
        reward = 0
        if 0 <= next_row < self.grid_size[0] and 0 <= next_col < self.grid_size[1]:
            # Move is valid


            self.agent_position = (next_row, next_col)


            if self.state[self.agent_position] != 1:
                reward += 1
            if self.state[self.agent_position] == 1:
                reward -= 1
            if self.state[self.agent_position] == 2:
                reward += 3
        else:
            # Move is invalid
            reward -= 1

        self.state[self.agent_position] = 1
            
        done = self.state.all() == 1
        if done:
            reward += 10

        return self.state, reward, done, {}