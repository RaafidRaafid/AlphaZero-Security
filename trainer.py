import torch
import torch.nn as nn
import numpy as np
from ResAlloc_env import gameEnv

class Representation:

    def __init__(self, representation):
        self.step_model = representation()

class ScorePredictionTrainer:

    def __init__(self, score_network, representation, env, learning_rate=0.05):
        self.step_model = score_network()
        self.representation = representation
        self.env = env
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
                    # print(' - gradient:', tensor.grad)
                    print(' - tens:', tensor)
                    # print()
                except AttributeError as e:
                    self.getBack(n[0])

        #print("chamatkar?")

    def train(self, states, feats, scores):
        edge_index = torch.LongTensor(self.env.edge_index)

        feat_mat = []
        for feat in feats:
            feat_mat.append(torch.FloatTensor(feat))
        feat_mat = torch.stack(feat_mat)

        self.optimizer.zero_grad()

        sts = []
        for state in states:
            state = torch.FloatTensor(state)
            sts.append(state)

        states = torch.stack(sts)
        scores = torch.FloatTensor(scores)

        outputs = self.step_model(states, feat_mat, edge_index)

        loss = self.value_criterion(scores.view(scores.size()[0],1), outputs)
        # loss = self.value_criterion(scores, outputs.view(-1))
        print(scores)
        print(outputs.view(-1))
        print(abs(scores - outputs.view(-1)))

        # logsoftmax = nn.LogSoftmax(dim=0)
        # loss = torch.mean(-scores*logsoftmax(outputs.view(-1)))
        # print(scores)
        # print(outputs.view(-1))
        # print(-scores*logsoftmax(outputs.view(-1)))

        loss.backward()
        self.optimizer.step()

        # self.getBack(loss.grad_fn)

        print("-------------------------------------------------------- ", loss)

class Trainer:

    def __init__(self, Policy, representation, env, type, learning_rate=0.02 ):

        self.step_model = Policy()
        self.representation = representation
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
        # inp = np.hstack((inp, np.zeros((inp.shape[0], 1), dtype=inp.dtype)))
        # for i in range(inp.shape[0]):
        #     inp[i][-1] = self.env.P_val[i]
        return inp

    def train(self, states, feats, search_pis, returns):

        edge_index = torch.LongTensor(self.env.edge_index)

        feat_mat = []
        for feat in feats:
            feat_mat.append(self.representation.step_model(torch.FloatTensor(feat), edge_index))
        feat_mat = torch.stack(feat_mat)

        value = []
        policy = []
        logits = []

        self.optimizer.zero_grad()

        search_pis = torch.FloatTensor(search_pis)
        returns = torch.FloatTensor(returns)


        sts = []
        for state in states:
            state = torch.FloatTensor(state)
            sts.append(state)

        states = torch.stack(sts)

        logits, y, z = self.step_model(states, feat_mat, edge_index)
        # for state in states:
        #     state = torch.FloatTensor(self.prep_input(state))
        #
        #     logit, y,z = self.step_model(state, edge_index)
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

        # print(abs(z-returns.view(returns.size()[0],1)))

        #print(loss_policy, loss_value)
        loss = 0.5*loss_policy + loss_value
        loss.backward()
        self.optimizer.step()

        # self.getBack(loss.grad_fn)

        # print("-------------------------------------------------------- ", 0.5*loss_policy, loss_value)

        return loss
