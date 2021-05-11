import numpy as np
from gym import core, spaces
from gym.envs.registration import register


class MDP(core.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(6)
        self.rng = np.random.RandomState(1234)
        self.reset()

    def reset(self):
        self.visited_six = False
        self.current_state = 1
        return self.current_state

    def step(self, action):
        if action == 0 or self.current_state == 5:
            self.current_state -= 1
        else: # action == 1
            self.current_state += self.rng.choice([-1, 1])

        done = False
        reward = 0
        if self.current_state == 0:
            done = True
            reward = 1.0 if self.visited_six else 0.01
        elif self.current_state == 5:
            self.visited_six = True

        return self.current_state, reward, done, None

class MDP_V1(core.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(6)
        self.rng = np.random.RandomState(1234)
        self.reset()

    def reset(self):
        self.visited_six = False
        self.current_state = 1
        return self.current_state

    def step(self, action):
        if action == 0:
            self.current_state -= 1
        elif action == 1 and not self.visited_six:
            self.current_state += self.rng.choice([-1, 1])
        else: # action == 1 and self.visited_six
            self.current_state += 1 if self.current_state < 5 else 0

        done = False
        reward = 0
        if self.current_state == 0:
            done = True
            reward = 1.0 if self.visited_six else 0.01
        elif self.current_state == 5:
            self.visited_six = True

        return self.current_state, reward, done, None



register(
    id='MDP-v0',
    entry_point='envs.mdp:MDP',
    reward_threshold=1,
)

register(
    id='MDP-v1',
    entry_point='envs.mdp:MDP_V1',
    reward_threshold=1,
)