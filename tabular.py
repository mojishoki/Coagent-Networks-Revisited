import gym
import numpy as np
from scipy.special import expit
from scipy.special import logsumexp


class SigmoidTermination:
    def __init__(self, rng, observation_space, lr, temp, device):
        assert(isinstance(observation_space, gym.spaces.Discrete))

        self.rng = rng
        self.weights = np.zeros((observation_space.n,))
        self.lr = lr
        self.temp = temp

    def pmf(self, observation):
        return expit(self.weights[observation] / self.temp)

    def sample(self, observation):
        return int(self.rng.uniform() < self.pmf(observation))

    def grad(self, observation, is_terminating):
        terminate = self.pmf(observation)
        return (1. - terminate) / self.temp if is_terminating else terminate/self.temp
    
    def update(self, observation, c, is_terminating):
        magnitude = self.grad(observation, is_terminating)
        self.weights[observation] += self.lr * magnitude * c


class SoftmaxPolicy:
    def __init__(self, rng, observation_space, nactions, temp):
        assert(isinstance(observation_space, gym.spaces.Discrete))

        self.rng = rng
        self.nactions = nactions
        self.weights = np.zeros((observation_space.n, self.nactions))
        self.temp = temp

    def pmf(self, observation):
        v = self.weights[observation]/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, observation):
        return int(self.rng.choice(self.nactions, p=self.pmf(observation)))
        


class SoftmaxQ(SoftmaxPolicy):
    def __init__(self, rng, observation_space, nactions, lr, temp, device):
        super().__init__(rng, observation_space, nactions, temp)
        # Here self.weights stores the Q-values as well
        self.lr = lr

    def update_Q(self, observation, action, target):
        self.weights[observation, action] += self.lr * (target - self.weights[observation, action])

    def value(self, observation, action=None):
        if action is None:
            return self.weights[observation, :]
        return self.weights[observation, action]


class SoftmaxAC(SoftmaxPolicy):
    def __init__(self, rng, observation_space, nactions, lr_actor, lr_critic, temp, device):
        super().__init__(rng, observation_space, nactions, temp)
        self.q = np.zeros((observation_space.n, nactions)) # Q-values
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
    def value(self, observation, action=None):
        if action is None:
            return self.q[observation, :]
        return self.q[observation, action]
    
    def update_pi(self, observation, action, c):
        actions_pmf = self.pmf(observation)
        self.weights[observation, :] -= self.lr_actor * c * actions_pmf  / self.temp
        self.weights[observation, action] += self.lr_actor * c  / self.temp

    def update_Q(self, observation, action, target):
        self.q[observation, action] += self.lr_critic * (target - self.q[observation, action])
