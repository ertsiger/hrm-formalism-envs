# Hierarchies of Reward Machines - Formalism & Environments
Implementation of the _formalism_ and the _environments_ described in the paper [Hierarchies of Reward Machines](#references).
The implementation of the policy and hierarchy learning algorithms can be found [here](https://github.com/ertsiger/hrm-learning).

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
   1. [CraftWorld Tasks](#craftworld-tasks)
   2. [WaterWorld Tasks](#waterworld-tasks)
4. [Citation](#citation)
5. [References](#references)

**Disclaimer:** In line with our previous work [(Furelos Blanco et al., 2021)](#references), we used the term _hierarchy
of subgoal automata_ instead of _hierarchy of reward machines_ during the initial stages of the work; hence, the code
employs the former term and the name of the package is `gym_hierarchical_subgoal_automata`.

## Overview
The environments in this repository are described in the [Hierarchies of Reward Machines paper](#references). In the
following lines, we describe some implementation decisions we have made in case you wish to extend the code or easily
spot the differences with respect to the codebases we build upon.

The CraftWorld environments are built on top of a modified version of 
[Minigrid](https://github.com/Farama-Foundation/Minigrid), which is installed from [this repository](https://github.com/ertsiger/hrm-minigrid).
Our modifications are exclusively done in the `gym_minigrid/minigrid.py` file:
* Added colors `white`, `pink`, `cyan` and `brown`. See the `COLORS` ([link](https://github.com/ertsiger/hrm-minigrid/tree/master/gym_minigrid/minigrid.py#L14)) and `COLOR_TO_IDX` ([link](https://github.com/ertsiger/hrm-minigrid/tree/master/gym_minigrid/minigrid.py#L30)) global variables.
* Added objects `iron`, `table`, `cow`, `sugarcane`, `wheat`, `chicken`, `redstone`, `rabbit`, `squid` and `workbench`.
See the `OBJECT_TO_IDX` ([link](https://github.com/ertsiger/hrm-minigrid/tree/master/gym_minigrid/minigrid.py#L46))
global variable and new object classes ([link](https://github.com/ertsiger/hrm-minigrid/tree/master/gym_minigrid/minigrid.py#L358)).

The WaterWorld code is based on the one by [Toro Icarte et al. (2018)](#references).

## Installation
The code has been tested using Python 3.7 on Linux and MacOS. We recommend to use a virtual environment since the 
requirements of this package may affect your current installation. To install the package, run the following commands:
```
$ cd hrm-formalism-envs
$ pip install -e .
```

To visualize the reward machines that compose a hierarchy, you should install 
[Graphviz](https://graphviz.org/download/):
```
# Ubuntu
$ sudo apt install graphviz

# OSX
$ brew install graphviz
```

## Usage
The repository contains implementations for different CraftWorld and WaterWorld tasks, as well as for the hierarchies
formalism we present in the aforementioned paper. The `example.py` and `test.py` files in the root of this repository
illustrate how environments are created and how the methods for traversing a hierarchy (among other things) are used.

To create an environment, you need to use the following Python code:
```
$ import gym, gym_hierarchical_subgoal_automata
$ env = gym.make(ENV_ID, params={"environment_seed": SEED})
```
where `ENV_ID` is the identifier of the environment, and `SEED` is an integer used as a seed to randomly initialize the
environment. Unless `random_restart: True`is specified inside the `params` dictionary, the environment will always be 
reset to the same initial state. Once the environment is created you can use `env.play()` to manually interact with it.

In the following sections we describe the identifiers of each task and the additional parameters that can be specified 
within the `params` dictionary.

### CraftWorld Tasks

#### Task Identifiers
The identifiers for the tasks used in the paper are:

| Task    | Id   |
|---------|------|
| Batter  | `CraftWorldBatter-v0` |
| Bucket  | `CraftWorldBucket-v0`  |
| Compass | `CraftWorldCompass-v0` |
| Leather | `CraftWorldLeather-v0` |
| Paper | `CraftWorldPaper-v0` |
| Quill | `CraftWorldQuill-v0` |
| Sugar | `CraftWorldSugar-v0` |
| Book | `CraftWorldBook-v0` |
| Map | `CraftWorldMap-v0` |
| MilkBucket | `CraftWorldMilkBucket-v0` |
| BookAndQuill | `CraftWorldBookAndQuill-v0` |
| MilkBucketAndSugar | `CraftWorldMilkBucketAndSugar-v0` |
| Cake | `CraftWorldCake-v0` |

#### Grid Types
The params for each type of grid used in the paper are given below. Note that the `grid_params` dictionary must be placed
within the `params` dictionary exemplified above, i.e. `env = gym.make(ENV_ID, params={"grid_params": {...}})`.

##### Open Plan (OP)
```
"grid_params": {
    "grid_type": "open_plan", "width": 7, "height": 7, "use_lava": False, "max_objs_per_class": 1
}
```

##### Open Plan + Lava (OPL)
```
"grid_params": {
    "grid_type": "open_plan", "width": 7, "height": 7, "use_lava": True, "num_lava": 1, "max_objs_per_class": 1
}
```

##### Four Rooms (FR)
```
"grid_params": {
    "grid_type": "four_rooms", "size": 13, "use_lava": False, "max_objs_per_class": 2
}
```

##### Four Rooms + Lava (FRL)
```
"grid_params": {
    "grid_type": "four_rooms", "size": 13, "use_lava": True, "max_objs_per_class": 2
}
```

#### Observation Format
The format the observations can be modified using the `state_format` parameter specified within the
`params` dictionary when creating the environment. There are three possible values:
* `tabular` - An integer representing the position in the grid.
* `one_hot` - Like `tabular` but in a one hot encoding.
* `full_obs` - The usual Minigrid observation *but* applied to the whole grid (i.e., not egocentric): one matrix
for the object ids, one for the color ids and one for the status of the objects.

### WaterWorld Tasks

#### Task Identifiers
The identifiers for the tasks used in the paper are:

| Task               | Id   |
|--------------------|------|
| RG                 | `WaterWorldRG-v0` |
| BC                 | `WaterWorldBC-v0`  |
| MY                 | `WaterWorldMY-v0` |
| RG&BC              | `WaterWorldRGAndBC-v0` |
| BC&MY              | `WaterWorldBCAndMY-v0` |
| RG&MY              | `WaterWorldRGAndMY-v0` |
| RGB                | `WaterWorldRGB-v0` |
| CMY                | `WaterWorldCMY-v0` |
| RGB&CMY            | `WaterWorldRGBAndCMY-v0` |

#### Scenario Types
There are two scenarios: without dead-ends (WOD) and with dead-ends (WD). Dead-ends are graphically represented by black
balls that must be avoided by the agent. By default, the environment has no dead-ends (i.e., it is WOD). To enable the
presence of dead-ends, you must enable the `avoid_black` flag inside the `params` dict, 
i.e. `env = gym.make(ENV_ID, params={"avoid_black": True, ...})`.

## Citation
If you use this code in your work, please use the following citation:
```
@inproceedings{FurelosBlancoLJBR23,
  author       = {Daniel Furelos-Blanco and
                  Mark Law and
                  Anders Jonsson and
                  Krysia Broda and
                  Alessandra Russo},
  title        = {{Hierarchies of Reward Machines}},
  booktitle    = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year         = {2023}
}
```
Remember to cite the original papers where the domains were proposed (see [Overview](#overview) for details).

## References
* Toro Icarte, R.; Klassen, T. Q.; Valenzano, R. A.; and McIlraith, S. A. 2018. [_Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning_](http://proceedings.mlr.press/v80/icarte18a.html). Proceedings of the 35th International Conference on Machine Learning (ICML). [Code](https://github.com/RodrigoToroIcarte/reward_machines).
* Furelos-Blanco, D.; Law, M.; Jonsson, A.; Broda, K.; and Russo, A. 2021. [_Induction and Exploitation of Subgoal Automata for Reinforcement Learning_](https://jair.org/index.php/jair/article/view/12372). Journal of Artificial Intelligence Research 70.
* Furelos-Blanco, D.; Law, M.; Jonsson, A.; Broda, K.; and Russo, A. 2023. [_Hierarchies of Reward Machines_](https://arxiv.org/abs/2205.15752). Proceedings of the 40th International Conference on Machine Learning (ICML).
