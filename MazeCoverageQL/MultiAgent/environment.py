import numpy as np

class GridWorld:
    def __init__(self, grid_size=(5, 5), POI_density=0.2):
        self.grid_size = grid_size
        self.POI_density = POI_density
        self.reward_grid = None
        self.coverage_grid = None
        self.reward_indices = self.get_POI()
        self.total_reward = 0
        self.position_grid = None
        self.observation_space = self.grid_size[0] * self.grid_size[1]
        self.action_space = 4
        self.total_steps = 0
        self.grid_visited = None


    def reset(self):
        self.coverage_grid = np.zeros(self.grid_size)
        self.reward_grid = self.place_POI()
        self.position_grid = np.zeros(self.grid_size)
        self.total_reward = 0
        self.total_steps = 0
        self.grid_visited = np.zeros((self.grid_size[0] + 1, self.grid_size[1] + 1))

    
    def get_POI(self):
        num_POI = int(self.POI_density * np.prod(self.grid_size))
        POI_indices = np.random.choice(np.prod(self.grid_size), num_POI, replace=False)
        POI_indices = np.unravel_index(POI_indices, self.grid_size)
        return POI_indices
    
    def place_POI(self):
        reward_grid = np.zeros(self.grid_size)
        reward_grid[self.reward_indices] = 2
        return reward_grid

    def step(self, action,agent):


        agent_row, agent_col = agent.position
        if action == 0:  # Up
            next_row, next_col = agent_row - 1, agent_col
        elif action == 1:  # Down
            next_row, next_col = agent_row + 1, agent_col
        elif action == 2:  # Left
            next_row, next_col = agent_row, agent_col - 1
        elif action == 3:  # Right
            next_row, next_col = agent_row, agent_col + 1
        reward = -0.05
        self.grid_visited[next_row,next_col] += 1

        if 0 <= next_row < self.grid_size[0] and 0 <= next_col < self.grid_size[1]:
            # Move is valid

            agent.position_grid[agent.position] = 0
            agent.position = (next_row, next_col)
            agent.position_grid[agent.position] = 1

            if self.coverage_grid[agent.position] != 1:
                reward += (self.reward_grid[agent.position] + 1) * 10
                reward += self.amount_covered() * self.observation_space

            if self.coverage_grid[agent.position] == 1:
                reward -= 1 * self.grid_visited[agent.position]


            self.reward_grid[agent.position] = 0
            self.coverage_grid[agent.position] = 1
            agent.coverage_grid[agent.position] = 1


        else:
            # Move is invalid
            reward -= 1 * self.grid_visited[next_row,next_col]
        done = self.coverage_grid.all() == 1
        if done:
            reward += 200 *  ((self.observation_space - 1) / agent.total_steps)
        

        self.total_reward += reward 
        agent.total_steps += 1
        agent.reward += reward
        return np.stack((agent.position_grid,self.coverage_grid,agent.other_agent_positions,self.reward_grid), axis= 2), reward, done, {}
    
    def get_other_agents(self,position):
        positions = self.position_grid.copy()
        positions[position] = 0
        return positions

    def update_agents(self,agents):
        for agent in agents:
            self.position_grid[agent.position] = 1
            self.coverage_grid[agent.position] = 1

    def get_state(self,agent):
        position = agent.position_grid
        other_positions = agent.other_agent_positions
        return np.stack((position,self.coverage_grid,other_positions,self.reward_grid), axis= 2)
    

    def amount_covered(self):
        return np.sum(self.coverage_grid) / self.observation_space
    
    def get_positions(self):
        [x,y] = np.where(self.coverage_grid == 1)



