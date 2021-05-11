import numpy as np
import torch
import random # just in case this thing is used elsewhere, to set the seed

from callback import Callback
from fire import Fire

import gym
from envs import fourrooms, mdp, mtcartpole, CartPoleSparse

import tabular, continuous


class FixedTermination:
    def __init__(self, terminate):
        self.p = 1 if terminate else 0

    def sample(self, observation):
        return self.p

    def pmf(self, observation):
        return self.p

class FixedActionPolicy:
    def sample(self, observation):
        return 0

###################################


class Option:
    def __init__(self, uid, policy, termination, children):
        self.uid = uid # a unique integers id
        self.policy = policy
        self.termination = termination
        self.is_primitive = isinstance(policy, FixedActionPolicy)
        self.children = children

        # If o is a non-primitve option and ch is one of its children: o.children[ch.uid] == ch
        if not self.is_primitive:
            self.children_indices = {ch.uid: i for i, ch in enumerate(children)}
        self.is_terminating = False
         
    def sample_policy(self, observation):
        index = self.policy.sample(observation)
        return self.children[index]

    def value(self, observation, action=None):
        if action:
            index = self.children_indices[action.uid]
            return self.policy.value(observation, index) # gives Q value for SoftmaxAC and SoftmaxQ
        else:
            return self.policy.value(observation) # gives list of Q values for softmaxAC and softmaxQ

    def update_Q(self, observation, action, tderror): # only makes sense if it's not a primitive option
        index = self.children_indices[action.uid]
        self.policy.update_Q(observation, index, tderror)

    def update_pi(self, observation, action, target): # only makes sense if it's neither primitive nor root
        index = self.children_indices[action.uid]
        self.policy.update_pi(observation, index, target)

    def sample_termination(self, observation):
        return self.termination.sample(observation)


class HOC:
    def __init__(self, rng, graph, env, discount, lr_intra, lr_critic, temperature_policy, lr_term, temperature_term, 
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
        
        self.reset()

    def reset(self):
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

            if i == len(self.graph)-1: # the root option
                p = self.RootPolicy(self.rng, self.env.observation_space, len(children), self.lr_critic, self.temperature_policy, self.device)
                t = FixedTermination(terminate=False)
            else:
                p = self.OptionPolicy(self.rng, self.env.observation_space, len(children), self.lr_intra, self.lr_critic, self.temperature_policy, self.device)
                t = self.Termination(self.rng, self.env.observation_space, self.lr_term, self.temperature_term, self.device)
            non_primitive_options.append(Option(counter, p, t, children=children))
            counter -= 1

        self.options = list(reversed(non_primitive_options)) + primitive_options
        self.root_option = self.options[0]
    
    def choose_primitive_option(self, observation, time, omega):

        child = omega.sample_policy(observation)

        while not omega.is_primitive:
            omega.last_observation = observation
            omega.active_child = child

            child.activation_time = time    # omega was active from before, child starts now
            child.reward = 0                # The total reward that child will accumulate while it's active
            child.active_parent = omega

            omega = child
            child = omega.sample_policy(observation)
        # now omega is a primitive option

        o = omega
        path_to_root = [o]
        while o is not self.root_option:
            o = o.active_parent
            path_to_root.append(o)
        # the path connecting the primitive option to the root option

        return omega, path_to_root

    
    def compute_local_r(self, j, path_to_root, observation, with_regularizer = True):
        r = 0
        coef = 1
        if with_regularizer:
            for oo in path_to_root[1:j+1]:
                coef *= oo.termination.pmf(observation)
        
        if j+2<len(path_to_root):
            r += coef * (1 - path_to_root[j+1].termination.pmf(observation)) * path_to_root[j+2].value(observation,path_to_root[j+1]) # target effect
            coef *= path_to_root[j+1].termination.pmf(observation)
        else:
            return coef * path_to_root[j+1].value(observation).max() # target effect
        
        for oo in path_to_root[j+2:]:
            if oo is not path_to_root[-1]:
                r += coef * (1 - oo.termination.pmf(observation)) * oo.active_parent.value(observation,oo) # generalized target effect
                coef *= oo.termination.pmf(observation)
            else:
                r += coef * (1 - oo.termination.pmf(observation)) * oo.value(observation).max()
                coef *= oo.termination.pmf(observation)
        
        return r
    
    def update_critics(self, observation, time, path_to_root, done):

        for j,o in enumerate(path_to_root):
            if not o.is_terminating:
                break
            r = self.compute_local_r(j, path_to_root, observation)
            p = o.active_parent
            update_target  = o.reward
            if not done:
                update_target += self.discount**(time - o.activation_time) * r
            p.update_Q(p.last_observation, o, update_target)

    def update_actors(self, observation, path_to_root, time, done):

        # Intra-option policy update (update the actor)
        for i, o in enumerate(path_to_root[:-2]):
            if not o.is_terminating:
                break
            
            p = path_to_root[i+1] # = o.active_parent
            critic_feedback = p.value(p.last_observation, o)
            if self.baseline:
                critic_feedback -= p.active_parent.value(p.last_observation, p) # target baseline
            p.update_pi(p.last_observation, o, critic_feedback)


    def update_terminations(self, observation, path_to_root, time):
        prob_list = [o.termination.pmf(observation) for o in path_to_root[1:]] # prob_list[-1] == 0, since the root never terminates
        q_list = [o.active_parent.value(observation) for o in path_to_root[1:-1]]
        index_list = [o.active_parent.children_indices[o.uid] for o in path_to_root[1:-1]]
        max_list = [max(q_list[i]) for i in range(len(q_list))] # max_list[i] == The maximum value that can be obtained by starting at path_to_root[i+2]
        
        k = 1
        while path_to_root[k].is_terminating:
            k += 1
        # path_to_root[k] is the one that has to choose a new option in the next step (the max value we can get from here is max_list[k-2])
        probs = 1
        for i, o in enumerate(path_to_root[:-2]):
            if not o.is_terminating:
                break
            p = o.active_parent
            if not p.is_terminating: # in this case the GAE A(s,notterminating) = sum rewards - V(s) = p.value(observation).max() - Q(s,p,u)
                diff = p.value(observation).max() - self.compute_local_r(i, path_to_root, observation, False)
                # diff = p.active_parent.value(observation, p) - self.compute_local_r(i, path_to_root, observation, False) #WITH TARGET ... kills the termination temperature effect
                val = diff * probs
            else: # in this case the GAE A(s,terminating) = sum rewards - V(s) = max_list[k-2] - Q(s,p,u)
                diff = max_list[k-2] - self.compute_local_r(i, path_to_root, observation, False)
                # diff = path_to_root[k].value(observation,path_to_root[k-1]) - self.compute_local_r(i, path_to_root, observation, False) #WITH TARGET ... kills the termination temperature effect
                val = - diff * probs # a negative one because it has terminated so -val is diff*probs which is GAE of A(s,terminating)

            p.termination.update(observation, -val, p.is_terminating)
            probs *= prob_list[i]


    def reset_history(self, nruns, nepisodes):
        n_total_options = len(self.graph) + self.env.action_space.n
        self.history = np.zeros((nruns, nepisodes, n_total_options, 3))
        # history[run, episode, option_uid, k]
        ## k == 0: avg duration
        ## k == 1: total discounted reward
        ## k == 2: number of times this option has been used
        # we consider all of the options, even primitive ones, this way we can also see how much each primitive option is used
        # and how much reward each of them have received (for primitive options, there is no discount)
        # For the root, this stores the duration and the total discounted reward of the entire episode


    def update_history(self, run, episode, time, omega):
        self.history[run, episode, omega.uid, 1] += omega.reward
        self.history[run, episode, omega.uid, 2] += 1
        self.history[run, episode, omega.uid, 0] += (1./ self.history[run, episode, omega.uid, 2]) * (time - omega.activation_time - self.history[run, episode, omega.uid, 0])
    
    def run(self, nruns, nepisodes, nsteps, callback=None):
        env = self.env

        self.reset_history(nruns, nepisodes)

        for run in range(nruns):
            
            self.reset()

            for episode in range(nepisodes):
                self.rng.choice(1) # for reproducibility
                observation = env.reset()
                observation = torch.tensor(observation, device = self.device)
                self.root_option.reward = 0
                self.root_option.activation_time = 0
                self.root_option.active_child = None
                omega, path_to_root = self.choose_primitive_option(observation, time=0, omega=self.root_option)

                for time in range(1, nsteps+1):
                    # now omega is primitive
                    observation, reward, done, _ = env.step(omega.sample_policy(observation))
                    observation = torch.tensor(observation, device = self.device)
                    # update rewards
                    for o in path_to_root:
                        o.reward += self.discount**(time-o.activation_time-1) * reward
                    
                    # termination might occur upon entering the new state
                    for o in path_to_root[:-1]:
                        if o.sample_termination(observation):
                            o.is_terminating = True
                        else:
                            break

                    self.update_critics(observation, time, path_to_root, done)
                    self.update_actors(observation, path_to_root, time, done)
                    self.update_terminations(observation, path_to_root, time)

                                        
                    if done:
                        for o in path_to_root:
                            o.is_terminating = False
                            self.update_history(run, episode, time, o)
                        break
                    else:
                        while omega.is_terminating:
                            omega.is_terminating = False
                            self.update_history(run, episode, time, omega)
                            omega = omega.active_parent


                    omega, path_to_root = self.choose_primitive_option(observation, time, omega)
                else: # when the agent didn't achieve goal so no break by done
                    for o in path_to_root:
                        self.update_history(run, episode, time, o)

                # if callback is passed on, call it at end of each episode
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


def main(game="Fourrooms-v0", discount=0.99, lr_term=1e-3, lr_intra=1e-5, lr_critic=1e-2, 
            nepisodes=250, nruns=100, nsteps=1000, graph=None, noptions=[2,2], shared=True, baseline=True, 
            temperature_policy=1e-2, temperature_term=1., seed=1, using_server=False, device = 'cpu'):

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
        Termination = tabular.SigmoidTermination
        OptionPolicy = tabular.SoftmaxAC
        RootPolicy = tabular.SoftmaxQ
    else: # continuous space discrete action
        Termination = continuous.SigmoidTermination
        OptionPolicy = continuous.SoftmaxAC
        RootPolicy = continuous.SoftmaxQ

    hoc = HOC(rng, graph, env, discount, lr_intra, lr_critic, temperature_policy, lr_term,
                temperature_term, baseline, OptionPolicy=OptionPolicy, RootPolicy=RootPolicy, Termination=Termination, 
                device = device)

    callback = Callback(__file__, args) # the object in charge of callbacks (saving, plotting, printing to console)
    hoc.run(nruns, nepisodes, nsteps, callback=callback) # the callable callback will be called at the end of each episode

    return hoc.history, hoc.options



if __name__ == '__main__':
    Fire(main)
