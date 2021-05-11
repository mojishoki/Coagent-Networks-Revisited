import numpy as np
from gym import core, spaces
from gym.envs.registration import register

class Fourrooms(core.Env):
    def __init__(self):
        self.layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in self.layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62
        self.init_states = list(range(self.observation_space.n))

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        self.goal = self.rng.choice(self.init_states)
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return state

    def newgoal(self,goal):
        self.goal = goal

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.

        We consider a case in which rewards are zero on all state transitions.
        """
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        done = state == self.goal
        return state, float(done), done, None



class FourroomsV1(core.Env): # The differenece with other versions is in the self.observation_space
    def __init__(self):
        self.layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in self.layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.spacesize = np.sum(self.occupancy == 0)

        self.observation_space = spaces.Box(low = np.zeros((self.spacesize,)), high = np.ones((self.spacesize,)))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62
        self.init_states = list(range(self.spacesize))

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        self.goal = self.rng.choice(self.init_states)
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        one_hot_state = np.zeros((self.spacesize,)  , dtype = np.float32 )
        one_hot_state[state] = 1.
        return one_hot_state

    def newgoal(self,goal):
        self.goal = goal

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.

        We consider a case in which rewards are zero on all state transitions.
        """
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        done = state == self.goal
        one_hot_state = np.zeros((self.spacesize,)  , dtype = np.float32 )
        one_hot_state[state] = 1.

        return one_hot_state, float(done), done, None



class FourroomsV2(core.Env): # The differenece with other versions is in the self.observation_space
    def __init__(self):
        self.layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in self.layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.spacesize = np.sum(self.occupancy == 0)

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([12.0, 12.0]))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62
        self.init_states = list(range(self.spacesize))

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        self.goal = self.rng.choice(self.init_states)
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return np.array(self.currentcell, dtype=np.float32)

    def newgoal(self,goal):
        self.goal = goal

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.

        We consider a case in which rewards are zero on all state transitions.
        """
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        done = state == self.goal
        return np.array(self.currentcell, dtype=np.float32), float(done), done, None



register(
    id='Fourrooms-v0',
    entry_point='envs.fourrooms:Fourrooms',
    reward_threshold=1,
)

register(
    id='Fourrooms-v1',
    entry_point='envs.fourrooms:FourroomsV1',
    reward_threshold=1,
)

register(
    id='Fourrooms-v2',
    entry_point='envs.fourrooms:FourroomsV2',
    reward_threshold=1,
)
