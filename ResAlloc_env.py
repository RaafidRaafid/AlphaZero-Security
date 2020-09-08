import numpy as np

from utils import *
from static_env import staticEnv

class gameEnv(staticEnv):

    init_alloc = None
    features = None
    out = None

    def __init__(self, id):
        self.adj, self.alloc, self.features, self.out = read_env_data("data/adj.txt", "data/node_info_" + str(id) + ".txt", "data/out_" + str(id) + ".txt")
        gameEnv.init_alloc = self.alloc
        gameEnv.features = self.features
        gameEnv.out = self.out
        self.degree = np.zeros(self.adj.shape[0], dtype=int)
        for i in range(self.adj.shape[0]):
            for j in range(self.adj.shape[1]):
                self.degree[i] += (self.adj[i][j] > 0)

        self.actions = []
        for i in range(self.adj.shape[0]):
            temp = []
            for j in range(self.adj.shape[1]):
                if self.adj[i][j] > 0.0:
                    temp.append((i,j))
            self.actions.append(temp)

        self.n_resources = 0
        for i in range(len(self.alloc)):
            if self.alloc[i] > 0:
                self.n_resources += 1
        self.n_nodes = self.adj.shape[0]

    def reset():
        pass

    # def step(self, action, type):
    #     if type == 'board':
    #         return self.alloc
    #     if self.alloc[action[0]] == 0 or self.alloc[action[1]] == 1:
    #         return None, None, None
    #     self.alloc[action[0]] = 0
    #     self.alloc[action[1]] = 1
    #     reward = self.features[action[1]] - self.features[action[0]]
    #
    #     return self.alloc, reward

    @staticmethod
    def next_state(alloc_state, action, type):
        if action == -1:
            return alloc_state
        temp_state = np.array([0.0]*len(alloc_state))
        temp_state[:] = alloc_state[:]
        if type == 'board':
            return temp_state
        if temp_state[action[0]] == 0 or temp_state[action[1]] == 1:
            return temp_state
        #print("amar dhon", temp_state[action[0]], temp_state[action[1]])
        temp_state[action[0]] = 0
        temp_state[action[1]] = 1
        #print(alloc_state)
        return temp_state

    @staticmethod
    def is_done_state(step_idx):
        return step_idx >= 30
        '''
        depth of the MCTS tree
        '''

    @staticmethod
    def initial_state():
        return gameEnv.init_alloc

    @staticmethod
    def get_obs_for_states(states):
        pass

    @staticmethod
    def get_return(alloc_state):
        before = 0.0
        after = 0.0
        for i in range(len(gameEnv.init_alloc)):
            before += gameEnv.init_alloc[i]*gameEnv.features[i]
            after += alloc_state[i]*gameEnv.features[i]
        return after-before

    @staticmethod
    def get_return_real(alloc_state):
        # print(np.sum(alloc_state*gameEnv.out) - np.sum(gameEnv.init_alloc*gameEnv.out))
        # return  np.sum(alloc_state*gameEnv.out) - np.sum(gameEnv.init_alloc*gameEnv.out)
        return np.sum(alloc_state*gameEnv.out)
