from abc import ABC, abstractmethod
from enum import Enum
import gym
from gym_hierarchical_subgoal_automata.automata.common import get_param
from gym_hierarchical_subgoal_automata.automata.hierarchical_automaton import HierarchicalAutomaton
from gym_hierarchical_subgoal_automata.automata.hierarchy import Hierarchy
from typing import Any, Dict, List, Set


class TaskEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class BaseEnv(ABC, gym.Env):
    ENVIRONMENT_RANDOM_SEED = "environment_seed"
    RANDOM_RESTART = "random_restart"

    USE_FLAT_HIERARCHY = "use_flat_hierarchy"

    # Processing of observations
    COMPRESS_OBS = "compress_obs"          # Whether to ignore an observation if its equal to the last one
    IGNORE_EMPTY_OBS = "ignore_empty_obs"  # Whether to ignore empty observations

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__()
        self.params = params
        self.seed = get_param(self.params, BaseEnv.ENVIRONMENT_RANDOM_SEED)
        self.random_restart = get_param(self.params, BaseEnv.RANDOM_RESTART, False)

        self.use_flat_hierarchy = get_param(params, BaseEnv.USE_FLAT_HIERARCHY, False)

        self.compress_obs = get_param(self.params, BaseEnv.COMPRESS_OBS, True)
        self.ignore_empty_obs = get_param(self.params, BaseEnv.IGNORE_EMPTY_OBS, True)

        self.hierarchy = None
        self.hierarchy_state = None

        self.last_obs = None

    def is_compressing_obs(self):
        return self.compress_obs

    def is_ignoring_empty_obs(self):
        return self.ignore_empty_obs

    def step(self, action):
        state, info = self.env_step(action)

        if self._is_obs_valid():
            self.hierarchy_state = self.hierarchy.get_next_hierarchy_state(self.hierarchy_state, self.get_observation())
            self.last_obs = self.get_observation()

        reward = 1.0 if self.is_goal_achieved() else 0.0
        return state, reward, self.is_terminal(), info

    def _is_obs_valid(self):
        obs = self.get_observation()
        if self.ignore_empty_obs and len(obs) == 0:
            return False
        if self.compress_obs and obs == self.last_obs:
            return False
        return True

    @abstractmethod
    def env_step(self, action):
        pass

    def is_terminal(self) -> bool:
        return self.hierarchy.is_terminal_state(self.hierarchy_state)

    def is_goal_achieved(self):
        return self.hierarchy.is_accept_state(self.hierarchy_state)

    @abstractmethod
    def get_observables(self) -> List[str]:
        pass

    @abstractmethod
    def get_restricted_observables(self) -> List[str]:
        pass

    @abstractmethod
    def get_observation(self) -> Set[str]:
        pass

    @abstractmethod
    def get_possible_observations(self) -> List[Set[str]]:
        """
        Returns all possible observations. This method should just be used for debugging purposes. It may also be used
        for the case in which the handcrafted machines are given to the agent to initialize the formula banks.
        """
        pass

    @abstractmethod
    def get_hierarchy(self) -> Hierarchy:
        pass

    def _build_hierarchy(self, root_automaton: HierarchicalAutomaton, dependency_hierarchies: List, dependency_automata: List):
        hierarchy = Hierarchy()
        hierarchy.set_root_automaton(root_automaton)
        for subhierarchy in dependency_hierarchies:
            for automaton_name in subhierarchy.get_automata_names():
                hierarchy.add_automaton(subhierarchy.get_automaton(automaton_name))
        for automaton in dependency_automata:
            hierarchy.add_automaton(automaton)
        return hierarchy

    def reset(self):
        self.last_obs = None
        if self.hierarchy is None:
            self.hierarchy = self.get_hierarchy()
        self.hierarchy_state = self.hierarchy.get_initial_state()
        return self.env_reset()

    @abstractmethod
    def env_reset(self):
        pass

    @abstractmethod
    def render(self, mode='human'):
        pass

    @abstractmethod
    def play(self):
        pass
