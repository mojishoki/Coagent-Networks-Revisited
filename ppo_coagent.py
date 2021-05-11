import gym
import numpy as np

import torch
import torch.nn as nn



class LinearNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.last_x = torch.tensor([], dtype = torch.float32)
        self.last_y = None

    def forward(self, x):
        if not self.training and torch.equal(self.last_x, x.cpu().float()):
            return self.last_y
        if not self.training:
            self.last_x = x.cpu().float()
            self.last_y = self.fc2(torch.relu(self.fc1(x.float())))
            return self.last_y
        else: # not training in which case no need to update last observations
            return self.fc2(torch.relu(self.fc1(x.float())))


    def initialize_zero(self, basenoise = .0):
        self.fc1.weight.data.fill_(basenoise)
        self.fc2.weight.data.fill_(basenoise)

    def initialize_xavier(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

class PPOCoAgent:
    def reset(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.impprobs = []
        self.act_time = []
        self.returns = []
        self.vs = []
        self.advs = []

    def ppo_advret(self, rewards):
        eps_len = len(rewards)

        for act_time in self.act_time:
            self.returns.append( np.sum( np.array([self.discount**i for i in range(eps_len-act_time)]) * np.array(rewards[act_time:])) )

        next_value = 0
        tmp_adv = 0
        act_time_ext = self.act_time + [eps_len]
        self.advs =[0.] * len(self.states)
        for i in reversed(range(len(self.states))):
            if i<len(self.states)-1: 
                next_value = self.vs[i+1]
            len_act_time = act_time_ext[i+1]-act_time_ext[i]
            rewards_i = np.sum( np.array([self.discount**k for k in range(len_act_time)]) * np.array(rewards[act_time_ext[i]:act_time_ext[i+1]]))
            tmp_td_error = rewards_i + self.discount**(len_act_time) * next_value - self.vs[i]
            tmp_adv = tmp_adv * self.eta * self.discount  + tmp_td_error
            self.advs[i] = tmp_adv

    def states_actions_to_prob(self, paths_to_root):
        raise NotImplementedError

    def ppo_update(self, paths_to_root):
        self.zero_grad()

        new_action_logprobs, new_vs, new_action_entropies = self.states_actions_to_prob(paths_to_root)

        if new_action_logprobs != []:
            self.train()
            surrogate = 0.
            ratio = (new_action_logprobs - torch.log(torch.tensor(self.action_probs)).to(self.device)).exp()
            ratio_clamped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            self.advs = torch.tensor(self.advs).to(self.device)
            Lsur = self.advs * ratio
            Lsur_clamped = self.advs * ratio_clamped
            Lsur_clipped = torch.min(Lsur, Lsur_clamped)
            value_loss=(new_vs - torch.tensor(self.returns).to(self.device))**2
            
            surrogate = torch.mean(-Lsur_clipped + self.c1 * value_loss - self.beta * new_action_entropies)
            surrogate.backward()
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.clip)
            self.step()
            self.eval()
            
    def update(self, rewards, paths_to_root):
        self.ppo_advret(rewards)
        for _ in range(self.SGD_epoch):
            self.ppo_update(paths_to_root)


class SigmoidTermination(PPOCoAgent):
    def __init__(self, rng, observation_space, discount, eta, hidden_dim, SGD_epoch, epsilon, c1, beta, clip, lr, temp, device):
        assert(isinstance(observation_space, gym.spaces.Box))

        self.rng = rng
        self.discount = discount
        self.eta = eta
        self.SGD_epoch = SGD_epoch
        self.epsilon = epsilon
        self.c1 = c1
        self.beta = beta
        self.clip = clip
        self.temp = temp
        self.device = device

        self.pi = LinearNN(np.prod(observation_space.shape), 1, hidden_dim).to(device)
        self.pi.initialize_zero(1e-3)
        self.optim = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.pi.eval()

    def train(self):
        self.pi.train()
    
    def eval(self):
        self.pi.eval()
    
    def zero_grad(self):
        self.optim.zero_grad()
    
    def step(self):
        self.optim.step()

    def pmf(self, observation):
        if self.pi.training:
            return torch.sigmoid(self.pi(observation)/self.temp)
        else:
            with torch.no_grad():
                return torch.sigmoid(self.pi(observation)/self.temp).item()

    def sample(self, observation, time):
        p = self.pmf(observation)
        idx =  int(self.rng.uniform() < p)
        prob = p if idx == 1 else 1 - p

        self.states.append(observation) 
        self.actions.append(idx)
        self.action_probs.append(prob)
        self.act_time.append(time)

        return idx, prob

    def value(self, observation, path_to_root_from_o):
        v = 0
        prod = 1.
        for p in path_to_root_from_o:
                v += prod * (1- p.termination.pmf(observation))*p.policy.value(observation) 
                prod *= p.termination.pmf(observation)
        return v

    def states_actions_to_prob(self, paths_to_root):
        new_action_logprobs = []
        new_action_entropies = []
        new_vs = []
        if self.states != []:
            self.train()

            for j,state in enumerate(self.states):
                o_termination_prob = self.pmf(state)
                new_action_entropies.append(-((1-o_termination_prob) * torch.log(1-o_termination_prob+1e-8) + o_termination_prob*torch.log(o_termination_prob+1e-8)))
                new_action_logprobs.append(torch.log(o_termination_prob if self.actions[j] else 1-o_termination_prob))
            new_action_entropies = torch.stack(new_action_entropies)
            new_action_logprobs = torch.stack(new_action_logprobs)

            for t,act_time in enumerate(self.act_time):
                path_to_root_act_time = paths_to_root[act_time-1]
                for oo_lev, oo in enumerate(path_to_root_act_time):
                    if oo.termination is self:
                        new_vs.append( self.value(self.states[t], path_to_root_act_time[oo_lev:]) )
                        break
            new_vs = torch.stack(new_vs)

            self.eval()
        return new_action_logprobs, new_vs, new_action_entropies



class SoftmaxAC(PPOCoAgent):
    def __init__(self, rng, observation_space, nactions, discount, eta, hidden_dim, SGD_epoch, epsilon, c1, beta, clip, lr_actor, lr_critic, temp, device):
        assert(isinstance(observation_space, gym.spaces.Box))

        self.rng = rng
        self.nactions = nactions
        self.discount = discount
        self.eta = eta
        self.SGD_epoch = SGD_epoch
        self.epsilon = epsilon
        self.c1 = c1
        self.beta = beta
        self.clip = clip
        self.temp = temp
        self.device = device

        self.pi = LinearNN(np.prod(observation_space.shape), self.nactions, hidden_dim).to(device)
        self.pi.initialize_zero()
        # self.pi.initialize_xavier()
        self.v = LinearNN(np.prod(observation_space.shape), 1, hidden_dim).to(device)
        self.v.initialize_zero()
        # self.v.initialize_xavier()

        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr_actor)
        self.v_optim  = torch.optim.Adam(self.v.parameters(), lr=lr_critic)
        self.pi.eval()
        self.v.eval()

    def train(self):
        self.pi.train()
        self.v.train()
    
    def eval(self):
        self.pi.eval()
        self.v.eval()
    
    def zero_grad(self):
        self.pi_optim.zero_grad()
        self.v_optim.zero_grad()
    
    def step(self):
        self.pi_optim.step()
        self.v_optim.step()


    def pmf(self, observation):
        if self.pi.training:
            return torch.softmax(self.pi(observation)/self.temp, dim=0)
        else:
            with torch.no_grad():
                return torch.softmax(self.pi(observation)/self.temp, dim=0).cpu().numpy()


    def sample(self, observation, time):
        p = self.pmf(observation)
        
        idx = int(self.rng.choice(self.nactions, p=p))
        prob = p[idx]
        
        self.impprobs.append(1.) # this will end up storing the probability of reaching here from the last primitive action
        self.states.append(observation)
        self.actions.append(idx)
        self.action_probs.append(prob)
        self.act_time.append(time)
        self.vs.append(self.value(observation))
        
        return idx, prob

    def value(self, observation, action=None):
        if action is None:
            if self.v.training:
                return self.v(observation)
            else:
                with torch.no_grad():
                    return self.v(observation).item()

    def states_actions_to_prob(self, paths_to_root=None):
        new_action_logprobs = []
        new_action_entropies = []
        new_vs = []
        if self.states != []:
            self.train()

            for j,state in enumerate(self.states):
                action_probs = self.pmf(state)
                new_action_entropies.append( -torch.sum(action_probs *torch.log(action_probs )) ) 
                new_action_logprobs.append( torch.log(self.pmf(state)[self.actions[j]]) )
            new_action_entropies = torch.stack(new_action_entropies)
            new_action_logprobs = torch.stack(new_action_logprobs)

            new_vs = torch.stack([self.value(state) for state in self.states])

            self.eval()

        return new_action_logprobs, new_vs, new_action_entropies
