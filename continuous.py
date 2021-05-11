import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 15)
        self.fc2 = nn.Linear(15, output_dim)

        self.last_x = torch.tensor([], dtype = torch.float32)
        self.last_y = None

    def forward(self, x):
        if not self.training and torch.equal(self.last_x, x.cpu().float()):
            return self.last_y
        self.last_x = x.cpu().float()
        self.last_y = self.fc2(torch.relu(self.fc1(x.float())))
        return self.last_y


    def initialize_zero(self):
        self.fc1.weight.data.fill_(.0)
        self.fc2.weight.data.fill_(.0)

    def initialize_xavier(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

class SigmoidTermination:
    def __init__(self, rng, observation_space, lr, temp, device):
        assert(isinstance(observation_space, gym.spaces.Box))

        self.rng = rng
        self.temp = temp

        self.model = LinearNN(np.prod(observation_space.shape), 1).to(device)
        # self.model.initialize_zero()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.eval()

    def pmf(self, observation):
        return F.sigmoid(self.model(observation)/self.temp).item()

    def sample(self, observation):
        return int(self.rng.uniform() < self.pmf(observation))

    def update(self, observation, c, is_terminating):
        self.model.train()
        self.optim.zero_grad()
        if is_terminating:
            loss = - F.logsigmoid(self.model(observation)/self.temp) * c.item()
        else:
            loss = F.logsigmoid(- self.model(observation)/self.temp) * c.item() 
        loss.backward()
        self.optim.step()
        self.model.eval()

class SoftmaxQ:
    def __init__(self, rng, observation_space, nactions, lr, temp, device):
        assert(isinstance(observation_space, gym.spaces.Box))

        self.rng = rng
        self.nactions = nactions
        self.temp = temp

        # Here self.model stores the Q-values as well
        self.q = LinearNN(np.prod(observation_space.shape), self.nactions).to(device)
        # self.q.initialize_zero()
        self.q_optim  = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.q.eval()

    def pmf(self, observation):
        return F.softmax(self.q(observation)/self.temp, dim=0).detach().cpu().numpy() 

    def sample(self, observation):
        p = self.pmf(observation)
        return int(self.rng.choice(self.nactions, p=p))

    def update_Q(self, observation, action, target):
        self.q.train()
        self.q_optim.zero_grad()
        loss = (self.q(observation)[action] - torch.tensor(target).data) ** 2
        loss.backward()
        self.q_optim.step()
        self.q.eval()

    def value(self, observation, action=None):
        if action is None:
            return self.q(observation)
        return self.q(observation)[action]


class SoftmaxAC:
    def __init__(self, rng, observation_space, nactions, lr_actor, lr_critic, temp, device):
        assert(isinstance(observation_space, gym.spaces.Box))

        self.rng = rng
        self.nactions = nactions
        self.temp = temp

        self.pi = LinearNN(np.prod(observation_space.shape), self.nactions).to(device)
        # self.pi.initialize_zero()
        # self.pi.initialize_xavier()

        self.q = LinearNN(np.prod(observation_space.shape), self.nactions).to(device)
        # self.q.initialize_zero()
        # self.q.initialize_xavier()

        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr_actor)
        self.q_optim  = torch.optim.Adam(self.q.parameters(), lr=lr_critic)
        self.pi.eval()
        self.q.eval()

    def pmf(self, observation):
        return F.softmax(self.pi(observation)/self.temp, dim=0).detach().cpu().numpy()

    def sample(self, observation):
        p = self.pmf(observation)
        return int(self.rng.choice(self.nactions, p=p))

    def value(self, observation, action=None):
        if action is None:
            return self.q(observation)
        return self.q(observation)[action]
    
    def update_pi(self, observation, action, c):
        self.pi.train()
        self.pi_optim.zero_grad()
        loss = - F.log_softmax(self.pi(observation)/self.temp, dim=0)[action] * c.item()
        loss.backward()
        self.pi_optim.step()
        self.pi.eval()

    def update_Q(self, observation, action, target):
        self.q.train()
        self.q_optim.zero_grad()
        loss = (self.q(observation)[action] - torch.tensor(target).data) ** 2
        loss.backward()
        self.q_optim.step()
        self.q.eval()