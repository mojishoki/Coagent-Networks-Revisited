# Coagent Networks Revisited
This repo contains code accompaning the paper, [Coagent Networks Revisited](https://arxiv.org/abs/). It includes code to run all the experiments described in the paper. For experiment details, please refer to the paper.

---

## Dependencies
To install dependencies for experiments, run the following commands:
``` bash
conda create -n hoc python=3.8
conda actvate hoc
pip install -r requirements.txt
```

## Usage
The main code is contained in `hoc.py`. The file `hppoc.py` contains an implementation of Hierarchical Proximal Policy Option-Critic (HPPOC), a generalization of PPOC.
The list of arguments may be seen in the `main` functions implemented in these files. 
Most of the arguments are self-explanatory.
Please refer to the paper for more details.

There are three arguments that may require some explanation.
The argument `graph` describes the graph of options with a list of lists of positive integers.
The elements of `graph` correspond to options and their elements are their list of children.
Any option having `-1` as one of its children will be capable of calling primitive options.
A primitive option is an options that executes its corresponding primitive action in the environment.
The following code checks if this variable is valid.
``` python
for i, v in enumerate(graph):
    assert(len(v) > 0)
    for j in v:
        # the option i is a parent of option j
        # if j == -1, then that option has all the primitive options as children as well as other j in v
        assert(j == -1 or (i < j and j < len(graph)))
```

If `graph` is `None`, then `noptions` and `shared` are used to generate the graph. 
The argument `noptions` is a list of positive integers.
If `shared` is `True`, then the children are shared.
In other words, we have a Feedforward Options Network.
Otherwise, we have a Hierarchical Option Critic.

For example, the following code will run the <1, 1, 1> FON which is analyzed in Figure 5 in the paper.
```bash
python hoc.py --noptions=[1,1] --seed 1 --nruns 5 --nepisodes 50000
```
This code will run the <1, 2, 2> FON model in the same figure.
```bash
python hoc.py --noptions=[2,2] --shared True --seed 1 --nruns 5 --nepisodes 50000
```
Here the root option has two children and each of them have two children, which they share.
Hence the root has 2 grand-children and the model has 5 non-primitive options.
If we set `shared False` in the above code, we will run an HOC with 7 non-primitive options.

In our experiments, we tend to use seeds from 1 to 5 or 1 to 10 and take 10 or 5 runs with 50000 episodes, for a total of 50 runs.

## Performance and Visualizations

An object of type `Callback` is called at the end of each episode.
At then end of each run, it will save both a copy of all the weights in the models and a history array.
This is a 4-dimensional array where
`history[run, episode, option_uid, k]`
is equal to:
- if k == 0: avg duration;
- if k == 1: total discounted reward;
- if k == 2: number of times this option has been used.

We consider all of the options, even primitive ones.
Using this we may analyse the options and compute several statics such as average option length.
For the root, this stores the duration and the total discounted reward of the entire episode, which is the performance of the model.

To see an example, check out the notebook `main.ipynb`.

---

## Related Works/Codes
* The fourrooms experiment is built on the Option-Critic, [2017](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14858/14328) [tabular code](https://github.com/jeanharb/option_critic/tree/master/fourrooms).
* The PPOC, [2017](https://arxiv.org/pdf/1712.00004.pdf) [code](https://github.com/mklissa/PPOC).
* Options of Interest, [2020](https://arxiv.org/abs/2001.00271), [code](https://github.com/kkhetarpal/ioc).

