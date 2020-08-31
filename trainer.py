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
        # print(var_grad_fn)
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], 'variable')
                    print("paisi", n[0], tensor.grad.size())
                    # print('Tensor with grad found:', tensor.size())
                    print(' - gradient:', tensor.grad)
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
        logits = []

        self.optimizer.zero_grad()

        search_pis = torch.FloatTensor(search_pis)
        returns = torch.FloatTensor(returns)

        adj = torch.FloatTensor(self.env.adj)

        sts = []
        for state in states:
            state = torch.FloatTensor(self.prep_input(state))
            sts.append(state)

        states = torch.stack(sts)

        logits, y, z = self.step_model(states, adj)
        # for state in states:
        #     state = torch.FloatTensor(self.prep_input(state))
        #
        #     logit, y,z = self.step_model(state, adj)
        #
        #     logits.append(logit.flatten())
        #     policy.append(y.flatten())
        #     value.append(z)

        logsoftmax = nn.LogSoftmax(dim=1)

        #print(search_pis, logsoftmax(logits), -search_pis*logsoftmax(logits))
        #print(search_pis, logits, search_pis-logits, logsoftmax(logits), -search_pis*logsoftmax(logits), torch.sum(-search_pis*logsoftmax(logits), dim=1))
        loss_policy = torch.mean(torch.sum(-search_pis*logsoftmax(logits), dim=1))
        #loss_policy = torch.mean(torch.sum(-logits*logsoftmax(search_pis), dim=1))
        loss_value = self.value_criterion(z, returns.view(returns.size()[0],1))

        #print(loss_policy, loss_value)
        loss = loss_policy + loss_value
        loss.backward()
        self.optimizer.step()

        #self.getBack(loss.grad_fn)

        print("-------------------------------------------------------- ", loss_policy, loss_value)

        return loss
