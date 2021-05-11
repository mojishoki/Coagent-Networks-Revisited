import numpy as np
import fire

import gym
from envs import fourrooms, mdp, mtcartpole, CartPoleSparse
import ppo_coagent
import torch
import random # just in case this thing is used elsewhere, to set seed
from callback import Callback

class FixedTermination:
    def __init__(self, terminate):
        self.p = 1 if terminate else 0
    def sample(self, observation, time):
        return self.p, 1
    def pmf(self, observation):
        return self.p

class FixedActionPolicy:
    def sample(self, observation, time):
        return 0, 1

###################################


class Option:
    def __init__(self, uid, policy, termination, children):
        self.uid = uid # a unique integers id
        self.policy = policy
        self.termination = termination
        self.is_primitive = isinstance(policy, FixedActionPolicy)
        self.is_root = isinstance(termination, FixedTermination) and not self.is_primitive
        self.children = children

        self.reset()

        # If o is a non-primitve option and ch is one of its children: o.children[ch.uid] == ch
        if not self.is_primitive:
            self.children_indices = {ch.uid: i for i, ch in enumerate(children)}
        self.is_terminating = False
         
    def sample_policy(self, observation, time):
        index, prob = self.policy.sample(observation, time)
        return self.children[index], prob

    def reset(self):
        if not self.is_primitive: 
            self.policy.reset()
            if not self.is_root:
                self.termination.reset()

    def update(self, rewards, paths_to_root):
        if not self.is_primitive: 
            self.policy.update(rewards, paths_to_root)
            if not self.is_root:
                self.termination.update(rewards, paths_to_root)

    def sample_termination(self, observation, time):
        return self.termination.sample(observation, time)


class HOC:
    def __init__(self, rng, graph, env, discount, lr_intra, lr_critic, hidden_dim, temperature_policy, 
                lr_term, temperature_term, eta , c1 , beta , epsilon, clip, SGD_epoch,
                baseline, OptionPolicy, RootPolicy, Termination, 
                device):

        for param, val in locals().items():
            if param != 'self':
                setattr(self, param, val)

        # check if graph is valid
        for i, v in enumerate(graph):
            assert(len(v) > 0)
            for j in v:
                # the option i is a parent of option j
                # if j == -1, then that option has all the primitive options as children as well as other j in v
                assert(j == -1 or (i < j and j < len(graph)))
        
        self.hard_reset()

    def reset(self):
        self.rewards = []
        self.paths_to_root = []
        for o in self.options:
            o.reset()

    def hard_reset(self):
        # make the options
        counter = len(self.graph) + self.env.action_space.n - 1
        primitive_options = []
        for a in range(self.env.action_space.n-1, -1, -1):
            p = FixedActionPolicy()
            t = FixedTermination(terminate=True)
            primitive_options.append(Option(counter, p, t, children=[a]))
            counter -= 1
        primitive_options = list(reversed(primitive_options))
        non_primitive_options = []
        for i, v in enumerate(reversed(self.graph)):
            children = [non_primitive_options[len(self.graph) - j - 1] for j in v if j >= 0]
            if -1 in v:
                children.extend(primitive_options)

            def get_ith(s, i):
                return s[i] if type(s)==list else s

            ppo_args = {
                "discount"  : get_ith(self.discount, i),
                "eta"       : get_ith(self.eta, i),
                "hidden_dim": get_ith(self.hidden_dim, i),
                "SGD_epoch" : self.SGD_epoch,
                "epsilon"   : self.epsilon,
                "c1"        : self.c1,
                "beta"      : self.beta,
                "clip"      : self.clip,
                "device"    : self.device,
            }
            policy_args = ppo_args.copy()
            policy_args.update({
                "lr_critic" : get_ith(self.lr_critic, i),
                "lr_actor"  : get_ith(self.lr_intra, i),
                "temp"      : get_ith(self.temperature_policy, i),
            })
            termination_args = ppo_args.copy()
            termination_args.update({
                "lr"        : get_ith(self.lr_term, i),
                "temp"      : get_ith(self.temperature_term, i),
            })

            if i == len(self.graph)-1: # the root option
                p = self.RootPolicy(self.rng, self.env.observation_space, len(children), **policy_args)
                t = FixedTermination(terminate=False)
            else:
                p = self.OptionPolicy(self.rng, self.env.observation_space, len(children), **policy_args)
                t = self.Termination(self.rng, self.env.observation_space, **termination_args)
            non_primitive_options.append(Option(counter, p, t, children=children))
            counter -= 1

        self.options = list(reversed(non_primitive_options)) + primitive_options
        self.root_option = self.options[0]

        self.reset()


    def choose_primitive_option(self, observation, time, omega):
        assert(not omega.is_primitive)
        
        child, prob = omega.sample_policy(observation, time)
        
        while not omega.is_primitive:
            old_prob = prob

            omega.last_observation = observation
            omega.active_child = child

            child.activation_time = time    # omega was active from before, child starts now
            child.reward = 0                # The total reward that child will accumulate while it's active
            child.active_parent = omega

            omega = child
            child, prob = omega.sample_policy(observation, time)
            if not omega.is_primitive:
                omega.policy.impprobs[-1] = omega.active_parent.policy.impprobs[-1] * old_prob # this will end up storing the probability of reaching here from the last primitive action
        # now omega is a primitive option

        o = omega
        path_to_root = [o]
        while o is not self.root_option:
            o = o.active_parent
            path_to_root.append(o)
        # the path connecting the primitive option to the root option

        return omega, path_to_root    

    
    
    def reset_history(self, nruns, nepisodes):
        n_total_options = len(self.graph) + self.env.action_space.n
        self.history = np.zeros((nruns, nepisodes, n_total_options, 3))
        # history[run, episode, option_uid, k]
        ## k == 0: avg duration
        ## k == 1: total discounted reward
        ## k == 2: number of times this option has been used
        # that we consider all of the options, even primitive ones, this way we can also see how much each primitive option is used
        # and how much reward each of them have received (for primitive options, there is no discount)
        # For the root, this stores the duration and the total discounted reward of the entire episode


    def update_history(self, run, episode, time, omega):
        self.history[run, episode, omega.uid, 1] += omega.reward
        self.history[run, episode, omega.uid, 2] += 1
        self.history[run, episode, omega.uid, 0] += (1./ self.history[run, episode, omega.uid, 2]) * (time - omega.activation_time - self.history[run, episode, omega.uid, 0])
    
    def run(self, nruns, nepisodes, nsteps, callback):
        env = self.env

        self.reset_history(nruns, nepisodes)

        for run in range(nruns):
            
            self.hard_reset()

            for episode in range(nepisodes):

                observation = env.reset()
                observation = torch.tensor(observation, device = self.device)
                self.root_option.reward = 0
                self.root_option.activation_time = 0
                self.root_option.active_child = None
                omega, path_to_root = self.choose_primitive_option(observation, time=0, omega=self.root_option)

                for time in range(1, nsteps+1):
                    # now omega is primitive
                    self.paths_to_root.append(path_to_root)
                    observation, reward, done, _ = env.step(omega.sample_policy(observation, time)[0])
                    self.rewards.append(reward)
                    observation = torch.tensor(observation, device = self.device)

                    # update rewards
                    for o in path_to_root:
                        if o is not self.root_option:
                            o.reward += o.active_parent.policy.discount**(time-o.activation_time-1) * reward
                        else:
                            o.reward += o.policy.discount**(time-o.activation_time-1) * reward

                    # first need to check if we are done or not
                    if done:
                        for o in path_to_root:
                            o.is_terminating = False
                            self.update_history(run, episode, time, o)
                        break

                    # termination might occur upon entering the new state
                    beta_probs = []
                    for o_lev, o in enumerate(path_to_root[:-1]):
                        terminating, prob =  o.sample_termination(observation, time)

                        beta_probs.append(prob)
                        if terminating:
                            o.is_terminating = True
                        else:
                            break
                    else: #when terminated all the way to the root
                        beta_probs.append(1.) #1,beta, ... , beta , 1
                    
                    n_terminated = len(beta_probs)
                    for o_lev,o in enumerate(path_to_root[1:n_terminated]):
                        if o is not self.root_option:
                            o.termination.vs.append( o.termination.value(observation, path_to_root[o_lev+1:]) )

                    beta_probs_cumprod = np.cumprod(beta_probs)
                    for i in range(1,n_terminated):
                        if path_to_root[i] is not self.root_option:
                            path_to_root[i].termination.impprobs.append(beta_probs_cumprod[i-1]) 

                    while omega.is_terminating:
                        omega.is_terminating = False
                        self.update_history(run, episode, time, omega)
                        omega = omega.active_parent

                    omega, path_to_root = self.choose_primitive_option(observation, time, omega)
                    
                    beta_probs = np.prod(beta_probs)
                    for i in range(1,n_terminated):
                        path_to_root[i].policy.impprobs[-1] *= beta_probs
                else: # when the agent didn't achieve goal so no break by done
                    for o in path_to_root:
                        self.update_history(run, episode, time, o)


                for o in self.options:
                    o.update(self.rewards, self.paths_to_root)

                self.reset()
                self.epsilon*=.999
                self.beta*=.995
                # If callback is not None passed on, call it an end of each episode
                if callback is not None:
                    callback(hoc=self, run=run, episode=episode, time=time)
            
        return self.history, self.options

###################################

def hoc_graph(noptions, shared):
    if noptions == []:
        return [[-1]]
    if shared:
        c = noptions[0]+1
        graph = [ list(range(1, c)) ]
        for i in range(len(noptions)-1):
            for _ in range(noptions[i]):
                graph.append(list(range(c, c + noptions[i+1])))
            c += noptions[i+1]
        graph.extend([-1] for _ in range(noptions[-1]))
    else:
        c = noptions[0]+1
        graph = [ list(range(1, c)) ]
        t = 1
        for i in range(len(noptions)-1):
            t *= noptions[i]
            for _ in range(t):
                graph.append(list(range(c, c + noptions[i+1])))
                c += noptions[i+1]
        t *= noptions[-1]
        graph.extend([-1] for _ in range(t))
    return graph


###################################


def main(game="Fourrooms-v0", discount=0.99, lr_term=1e-3, lr_intra=1e-5, lr_critic=1e-2, hidden_dim = 15, 
            eta = 0.95, c1 = 0.5, beta = 0.01 ,epsilon = 0.2 ,clip = 0.1 , SGD_epoch = 20,
            nepisodes=250, nruns=100, nsteps=1000, graph=None, noptions=[2,2], shared=True, baseline=True, 
            temperature_policy=1e-2, temperature_term=1., seed=1, using_server=False,
            device = 'cpu'):

    args = dict(locals().items())

    rng = np.random.RandomState(seed)
    env = gym.make(game)
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if graph is None:
        if not isinstance(noptions, list):
            noptions = eval(noptions)
            assert(isinstance(noptions, list))
        graph = hoc_graph(noptions, shared=shared)
    elif not isinstance(graph, list):
        graph = eval(graph)
        assert(isinstance(graph, list))

    if isinstance(env.observation_space, gym.spaces.Discrete):
        print('ERROR: Only continuous observation space allowed')
        return
    else: # continuous space discrete action
        Termination = ppo_coagent.SigmoidTermination
        OptionPolicy = ppo_coagent.SoftmaxAC
        RootPolicy = ppo_coagent.SoftmaxAC

    hoc = HOC(rng, graph, env, discount, lr_intra, lr_critic, hidden_dim, temperature_policy, lr_term, temperature_term, eta , c1 , beta, epsilon, clip, SGD_epoch,
                baseline, OptionPolicy=OptionPolicy, RootPolicy=RootPolicy, Termination=Termination, 
                device = device)

    callback = Callback(__file__, args) # the object in charge of callbacks (saving, plotting, printing to console)
    hoc.run(nruns, nepisodes, nsteps, callback=callback) # the callable callback will be called at the end of each episode

    return hoc.history, hoc.options



if __name__ == '__main__':
    fire.Fire(main)
