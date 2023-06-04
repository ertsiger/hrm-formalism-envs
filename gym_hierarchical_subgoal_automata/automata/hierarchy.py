from gym_hierarchical_subgoal_automata.automata.common import HierarchyState
from gym_hierarchical_subgoal_automata.automata.condition import FormulaCondition
from gym_hierarchical_subgoal_automata.automata.hierarchical_automaton import HierarchicalAutomaton
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula, TRUE
from typing import Dict, Optional, Set, Tuple


class Hierarchy:
    def __init__(self):
        self._root_automaton_name: Optional[str] = None
        self._automata: Dict[str, HierarchicalAutomaton] = {}

    def get_root_automaton(self):
        return self.get_automaton(self._root_automaton_name)

    def get_root_automaton_name(self):
        return self._root_automaton_name

    def set_root_automaton(self, automaton: HierarchicalAutomaton):
        self.add_automaton(automaton)
        self._root_automaton_name = automaton.get_name()

    def add_automaton(self, automaton: HierarchicalAutomaton):
        self._automata[automaton.get_name()] = automaton

    def get_automaton(self, automaton_name: str) -> HierarchicalAutomaton:
        return self._automata[automaton_name]

    def get_automata_names(self):
        return self._automata.keys()

    def get_initial_state(self) -> HierarchyState:
        root_automaton = self.get_automaton(self._root_automaton_name)
        return root_automaton.get_initial_hierarchy_state(TRUE, [])

    def get_next_hierarchy_state(self, hierarchy_state: HierarchyState, observation: Set[str]) -> HierarchyState:
        # copy the state but with an empty list of satisfied calls
        next_hierarchy_state = HierarchyState(hierarchy_state.state_name, hierarchy_state.automaton_name,
                                              hierarchy_state.context, hierarchy_state.stack, [])
        automaton = self.get_automaton(hierarchy_state.automaton_name)
        return automaton.get_next_hierarchy_state(next_hierarchy_state, observation, self)

    def is_accept_state(self, hierarchy_state: HierarchyState):
        """Tells whether a given state is the accepting state."""
        return hierarchy_state.automaton_name == self._root_automaton_name and \
               self._automata[self._root_automaton_name].is_accept_hierarchy_state(hierarchy_state)

    def is_reject_state(self, hierarchy_state: HierarchyState):
        """Tells whether a given state is a rejecting state."""
        return self._automata[hierarchy_state.automaton_name].is_reject_hierarchy_state(hierarchy_state)

    def is_terminal_state(self, hierarchy_state: HierarchyState):
        return self.is_accept_state(hierarchy_state) or self.is_reject_state(hierarchy_state)

    def is_root_automaton(self, automaton_name):
        return automaton_name == self._root_automaton_name

    def get_subgoals(self, subgoals: Set[FormulaCondition], ignore_rejecting_transitions=False, only_satisfiable=False):
        """Fills the set of subgoals with the conjunctive formulas that appear in the hierarchy."""
        self.get_root_automaton().get_hierarchy_subgoals(subgoals, TRUE, self, ignore_rejecting_transitions,
                                                         only_satisfiable)

    def get_hierarchy_states(self, hierarchy_states: Set[Tuple[str, str, DNFFormula]]):
        """
        Returns all possible (automaton, automaton state, context) tuples in the hierarchy. Note that the term
        'hierarchy state' is overloaded (we do not return all possible stacks) but used for simplicity.
        """
        self.get_root_automaton().get_hierarchy_states(hierarchy_states, TRUE, self)

    def is_deterministic(self) -> bool:
        raise NotImplementedError
