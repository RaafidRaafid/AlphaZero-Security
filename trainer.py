import torch
import torch.nn as nn
import numpy as np
from ResAlloc_env import gameEnv

class Trainer:

    def __init__(self, Policy, env, type, learning_rate=0.1):

        self.step_model = Policy()
        self.env = env
        self.type = type
        self.learning_rate = learning_rate

        self.value_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.step_model.parameters(),lr=self.learning_rate)

    def getBack(self, var_grad_fn):
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], 'variable')
                    # print("paisi", n[0])
                    # print('Tensor with grad found:', tensor)
                    # print(' - gradient:', tensor.grad)
                    # print()
                except AttributeError as e:
                    self.getBack(n[0])

        #print("chamatkar?")

    def prep_input(self, state):
        r = []
        for i in range(len(state)):
            if state[i] > 0:
                r.append(i)
        inp = np.zeros((len(state), len(r)))
        for i in range(len(r)):
            inp[r[i]][i] = 1.0
        inp = np.hstack((inp, np.zeros((inp.shape[0], 1), dtype=inp.dtype)))
        for i in range(inp.shape[0]):
            inp[i][-1] = self.env.P_val[i]
        return inp

    def train(self, states, search_pis, returns):

        value = []
        policy = []

        self.optimizer.zero_grad()


        search_pis = torch.FloatTensor(search_pis)
        returns = torch.FloatTensor(returns)

        adj = torch.FloatTensor(self.env.adj)
        for state in states:
            state = torch.FloatTensor(self.prep_input(state))

            y,z = self.step_model(state, adj)

            policy.append(y.flatten())
            value.append(z)

        policy = torch.cat(policy, 0)
        value = torch.cat(value, 0)

        #print(policy, search_pis)

        logsoftmax = nn.LogSoftmax(dim=1)
        loss_policy = self.value_criterion(search_pis.flatten(), policy)
        loss_value = self.value_criterion(value, returns)

        #print(loss_policy, loss_value)
        loss = loss_policy + loss_value
        loss.backward()
        self.optimizer.step()
        #print(loss_value.grad_fn)

        print("-------------------------------------------------------- ", loss_policy, loss_value)

        return loss
