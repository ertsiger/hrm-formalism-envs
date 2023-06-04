from collections import deque
from gym_hierarchical_subgoal_automata.automata.condition import CallCondition, EdgeCondition, FormulaCondition
from gym_hierarchical_subgoal_automata.automata.common import CallStackItem, HierarchyState, SatisfiedCall, \
    SatisfiedCallContext
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula, TRUE
import numpy as np
import os
import subprocess
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple


class NonDeterminismException(Exception):
    def __init__(self):
        super().__init__("Error: The automaton is not deterministic.")


class SatisfiedTransition(NamedTuple):
    from_state_name: str
    to_state_name: str
    automaton_name: str


class HierarchicalAutomaton:
    GRAPHVIZ_SUBPROCESS_TXT = "diagram.txt"  # name of the file temporally used to store the automata in Graphviz format
    
    def __init__(self, name):
        self._name = name
        self._states: List[str] = []
        self._edges: Dict[str, Dict[str, List[EdgeCondition]]] = {}
        self._initial_state: Optional[str] = None
        self._accept_state: Optional[str] = None
        self._reject_state: Optional[str] = None

    def get_name(self):
        return self._name

    def add_state(self, state: str):
        """Adds a state to the set of states and creates an entry in the set of edges that go from that state."""
        if state not in self._states:
            self._states.append(state)
            self._states.sort()
            self._edges[state] = {}

    def get_states(self):
        """Returns the set of states."""
        return self._states

    def get_num_states(self):
        """Returns the number of states in the automaton."""
        return len(self._states)

    def get_state_embedding(self, state: str):
        """Returns a one-hot vector representation of a state in the automaton."""
        states = sorted(self._states)
        state_e = np.zeros(self.get_num_states(), dtype=np.float32)
        state_e[states.index(state)] = 1.0
        return state_e

    def get_edges(self):
        return self._edges

    def get_num_edges(self):
        return sum([self.get_num_outgoing_conditions(state) for state in self._states])

    def add_formula_edge(self, from_state: str, to_state: str, formula: DNFFormula):
        self._add_edge_common(from_state, to_state, FormulaCondition(formula))

    def add_call_edge(self, from_state: str, to_state: str, called_automaton: str, formula: DNFFormula):
        self._add_edge_common(from_state, to_state, CallCondition(called_automaton, formula))

    def _add_edge_common(self, from_state: str, to_state: str, condition: EdgeCondition):
        """
        Adds an edge to the set of edges. Note that the set of edges is (from_state) -> list((condition, to_state)).
        If the states are not in the set of states, they are added.
        """
        self._add_undefined_states_in_edges(from_state, to_state)
        if to_state not in self._edges[from_state]:
            self._edges[from_state][to_state] = []
        self._edges[from_state][to_state].append(condition)

    def _add_undefined_states_in_edges(self, from_state: str, to_state: str):
        if from_state not in self._edges:
            self.add_state(from_state)
        if to_state not in self._edges:
            self.add_state(to_state)

    def get_num_outgoing_conditions(self, state: str, context: DNFFormula = TRUE):
        return len(self.get_outgoing_conditions(state, context))

    def get_outgoing_conditions(self, from_state: str, context: DNFFormula = TRUE) -> List[Tuple[EdgeCondition, str]]:
        """
        Returns one condition for each disjunct in the context of a call from a given state, and one condition for each
        disjunct in the DNF formula that results from appending the context to the outgoing DNF formula.
        """
        conditions = []
        for to_state in self._edges[from_state]:
            for condition in self._edges[from_state][to_state]:
                if condition.is_call():
                    for dec_condition in condition.decompose():
                        conditions.append((dec_condition, to_state))
                else:
                    dnf_formula = condition.get_formula().logic_and(context)
                    for dnf_disjunct in dnf_formula.get_dnf_disjuncts():
                        conditions.append((FormulaCondition(dnf_disjunct), to_state))
        return conditions

    def get_rejecting_outgoing_conditions_mask(self, automaton_state):
        return [1 if (c, self.get_reject_state()) in self.get_outgoing_conditions(automaton_state) else 0
                for c in self.get_automaton_subgoals()]

    def _get_outgoing_conditions_criteria(self, from_state: str, context: DNFFormula, observations, hierarchy,
                                          found_state_criteria, include_context):
        """
        Returns outgoing conditions from a given state that lead to paths that can reach a state that satisfies the passed
        criteria in the automaton (not in the hierarchy). In other words, the automaton can be traversed such that it is
        guaranteed that a state satisfying the crtieria can be reached in the automaton from that state and given a set
        of the observations that the agent might see while interacting with the environment.
        """
        conditions = []
        for to_state in self._edges[from_state]:
            for condition in self._edges[from_state][to_state]:
                if condition.is_call():
                    for dec_condition in condition.decompose():
                        called_automaton = hierarchy.get_automaton(condition.get_called_automaton())
                        new_context = context.logic_and(dec_condition.get_context())

                        if called_automaton.is_satisfiable(new_context, observations, hierarchy):
                            # If we take any transition the context becomes TRUE.
                            if self._is_state_reachable(to_state, TRUE, observations, hierarchy, found_state_criteria):
                                conditions.append((dec_condition, to_state))
                else:
                    for cond_disjunct in condition.get_formula().get_dnf_disjuncts():
                        for ctx_disjunct in context.get_dnf_disjuncts():
                            joint_formula = cond_disjunct.logic_and(ctx_disjunct)
                            if joint_formula.is_satisfiable() and any(joint_formula.is_satisfied(obs) for obs in observations):
                                # If we take any transition the context becomes TRUE.
                                if self._is_state_reachable(to_state, TRUE, observations, hierarchy, found_state_criteria):
                                    if include_context:
                                        conditions.append((FormulaCondition(joint_formula), to_state))
                                    else:
                                        conditions.append((FormulaCondition(cond_disjunct), to_state))
        return conditions

    def get_outgoing_conditions_with_terminating_paths(self, from_state: str, context: DNFFormula, observations, hierarchy,
                                                       include_context):
        return self._get_outgoing_conditions_criteria(from_state, context, observations, hierarchy, self.is_terminal_state,
                                                      include_context)

    def is_satisfiable(self, context: DNFFormula, observations: Set, hierarchy):
        """
        Returns True if there is a path from the initial state to the accepting state such that a set of passed
        observations is satisfied. Else, returns False. The satisfiability of the automaton is examined under context
        TRUE. Note that, therefore, cycles are not worth examining and we can perform just a normal search.

        We check whether any of the outgoing edges from the initial state of the called automata is satisfied taking
        the context into account. If so, then we can check for the whole satisfiability of the automaton without that
        context.
        """
        return self._is_state_reachable(self.get_initial_state(), context, observations, hierarchy, self.is_accept_state)

    def _is_state_reachable(self, state: str, context: DNFFormula, observations: Set, hierarchy, found_state_criteria: Callable[[str], bool]):
        # If the state is not the initial state, then the context must be TRUE.
        assert self.is_initial_state(state) or (not self.is_initial_state(state) and context.is_true())

        visited = set()
        state_queue = deque([(state, context)])

        if found_state_criteria(state):
            return True

        while len(state_queue) > 0:
            current_state, current_context = state_queue.popleft()
            visited.add((current_state, current_context))

            for to_state in self._edges[current_state]:
                for condition in self._edges[current_state][to_state]:
                    if condition.is_call():
                        is_condition_sat = hierarchy.get_automaton(condition.get_called_automaton()).is_satisfiable(
                            condition.get_context().logic_and(context),
                            observations,
                            hierarchy
                        )
                    else:
                        joint_formula = condition.get_formula().logic_and(context)
                        is_condition_sat = joint_formula.is_satisfiable() and \
                                           any(joint_formula.is_satisfied(obs) for obs in observations)

                    if is_condition_sat:
                        if found_state_criteria(to_state):
                            return True
                        new_state_ctx = (to_state, TRUE)  # Any transition leads to have context TRUE
                        if new_state_ctx not in visited:
                            state_queue.append(new_state_ctx)

        return False

    def get_called_automaton_names(self):
        """
        Returns a set containing the names of the automata called from this automaton.
        """
        automaton_names = set()
        for from_state in self._states:
            for to_state in self._edges[from_state]:
                for condition in self._edges[from_state][to_state]:
                    if condition.is_call():
                        automaton_names.add(condition.get_called_automaton())
        return automaton_names

    def has_incoming_transitions(self, to_state: str):
        """
        Returns True if the state passed as a parameter has an incoming transition from any other state in the
        automaton. Else, returns False.
        """
        for from_state in self._states:
            if to_state in self._edges[from_state]:
                return True
        return False

    def get_hierarchy_states(self, hierarchy_states: Set[Tuple[str, str, DNFFormula]], context: DNFFormula, hierarchy):
        for state in self.get_non_deadend_states():
            if self.is_initial_state(state):
                for dnf_disjunct in context.get_dnf_disjuncts():
                    hierarchy_states.add((self._name, state, dnf_disjunct))
                if self.has_incoming_transitions(state):
                    # if there are any incoming transitions, the accumulated context is lost
                    hierarchy_states.add((self._name, state, TRUE))
            else:
                hierarchy_states.add((self._name, state, TRUE))

            outgoing_edges = self._edges[state]
            for to_state in outgoing_edges:
                for call_condition in [cond for cond in outgoing_edges[to_state] if cond.is_call()]:
                    called_automaton = hierarchy.get_automaton(call_condition.get_called_automaton())
                    if self.is_initial_state(state):
                        new_context = context.logic_and(call_condition.get_context())
                        called_automaton.get_hierarchy_states(hierarchy_states, new_context, hierarchy)
                        if self.has_incoming_transitions(state):
                            called_automaton.get_hierarchy_states(hierarchy_states, call_condition.get_context(), hierarchy)
                    else:
                        called_automaton.get_hierarchy_states(hierarchy_states, call_condition.get_context(), hierarchy)

    def get_edge_id_from_state_for_condition(self, state: str, condition: EdgeCondition, context: DNFFormula = TRUE):
        """
        Returns the position of a given condition in the array of outgoing conditions from a state given a context.
        """
        conditions, _ = zip(*self.get_outgoing_conditions(state, context))
        return conditions.index(condition)

    def is_deadend_state(self, state: str):
        """
        Indicates whether a given state has outgoing transitions or not.
        """
        return self.get_num_outgoing_conditions(state) == 0

    def get_non_deadend_states(self):
        """
        Returns a list containing the states with at least one outgoing transition.
        """
        return [s for s in self._states if not self.is_deadend_state(s)]

    def set_initial_state(self, state: str):
        """Sets the initial state (there can only be one initial state)."""
        self._initial_state = state

    def get_initial_state(self):
        """Returns the name of the initial state."""
        return self._initial_state

    def is_initial_state(self, state: str):
        """Tells whether a given state is the initial state."""
        return state == self._initial_state

    def get_initial_hierarchy_state(self, context: DNFFormula, stack: List[CallStackItem]) -> HierarchyState:
        return HierarchyState(self._initial_state, self._name, context, stack, [])

    def set_accept_state(self, state: str):
        """Sets a given state as the accepting state."""
        self._accept_state = state

    def get_accept_state(self):
        """Returns the name of the accepting state."""
        return self._accept_state

    def is_accept_state(self, state: str):
        """Tells whether a given state is the accepting state."""
        return state == self._accept_state

    def is_accept_hierarchy_state(self, state: HierarchyState) -> bool:
        return self.is_accept_state(state.state_name)

    def has_accept_state(self):
        """Returns true if the automaton contains an accepting state."""
        return self._accept_state is not None

    def set_reject_state(self, state: str):
        """Sets a given state as the rejecting states."""
        self._reject_state = state

    def get_reject_state(self):
        """Returns the name of the rejecting state."""
        return self._reject_state

    def is_reject_state(self, state: str):
        """Tells whether a given state is the rejecting state."""
        return state == self._reject_state

    def is_reject_hierarchy_state(self, state: HierarchyState) -> bool:
        return self.is_reject_state(state.state_name)

    def has_reject_state(self):
        """Returns true if the automaton contains a rejecting state."""
        return self._reject_state is not None

    def is_terminal_state(self, state: str):
        """Tells whether a given state is terminal (accepting or rejecting state)."""
        return self.is_accept_state(state) or self.is_reject_state(state)

    def get_next_hierarchy_state(self, hierarchy_state: HierarchyState, observation: Optional[Set[str]], hierarchy) -> HierarchyState:
        assert hierarchy_state.automaton_name == self._name
        assert (hierarchy_state.automaton_name == hierarchy.get_root_automaton_name() and hierarchy_state.is_stack_empty()) \
                or hierarchy_state.automaton_name != hierarchy.get_root_automaton_name()

        if self.is_accept_hierarchy_state(hierarchy_state):
            if not hierarchy_state.is_stack_empty():
                # return control to calling automata (which performs the transition given by their calls)
                stack_top = hierarchy_state.get_stack_top()
                next_hierarchy_state = \
                    HierarchyState(stack_top.to_state_name, stack_top.automaton_name, TRUE,  # the context is lost after the call is sat
                                   hierarchy_state.get_substack(),
                                   hierarchy_state.satisfied_calls + [SatisfiedCall(stack_top.automaton_name, stack_top.call_condition)])
                next_automaton = hierarchy.get_automaton(stack_top.automaton_name)
                return next_automaton.get_next_hierarchy_state(next_hierarchy_state, observation, hierarchy)
        elif observation is not None:  # None = "no observation to process" used for returning calls
            next_hierarchy_state = self.get_next_hierarchy_state_helper(hierarchy_state, observation, hierarchy)
            if next_hierarchy_state is not None:
                # if new state is an accepting state we need to return control to the calling state in the stack
                next_automaton = hierarchy.get_automaton(next_hierarchy_state.automaton_name)
                return next_automaton.get_next_hierarchy_state(next_hierarchy_state, None, hierarchy)
        return hierarchy_state

    def get_next_hierarchy_state_helper(self, hierarchy_state: HierarchyState, observation: Set[str], hierarchy) -> Optional[HierarchyState]:
        assert self.get_name() == hierarchy_state.automaton_name

        next_hierarchy_states = []
        self._get_next_hierarchy_states_from_sat_transitions(next_hierarchy_states, hierarchy_state, observation)
        self._get_next_hierarchy_sates_from_calls(next_hierarchy_states, hierarchy_state, observation, hierarchy)

        if len(next_hierarchy_states) > 1:
            raise NonDeterminismException
        elif len(next_hierarchy_states) == 1:
            return next_hierarchy_states.pop()
        return None

    def _get_next_hierarchy_states_from_sat_transitions(self, next_hierarchy_states: List[HierarchyState],
                                                        hierarchy_state: HierarchyState, observation: Set[str]):
        """
        Returns the hierarchy states resulting from the satisfied transitions from the hierarchy state passed as a
        parameter. Note that they are put in a set to avoid repeating satisfied transitions in case two formulas in an
        OR are simultaneously satisfied; this is important later when checking if determinism holds).
        """
        sat_transitions = {SatisfiedTransition(hierarchy_state.state_name, to_state, self._name)
                           for to_state in self._edges[hierarchy_state.state_name]
                           for condition in self._edges[hierarchy_state.state_name][to_state]
                           if not condition.is_call() and condition.is_satisfied(observation)}
        for transition in sat_transitions:
            next_hierarchy_states.append(HierarchyState(transition.to_state_name, transition.automaton_name, TRUE,
                                                        hierarchy_state.stack, []))

    def _get_next_hierarchy_sates_from_calls(self, next_hierarchy_states: List[HierarchyState],
                                             hierarchy_state: HierarchyState, observation: Set[str], hierarchy):
        call_edges = [SatisfiedCallContext(condition.get_satisfied_condition(observation), to_state)
                      for to_state in self._edges[hierarchy_state.state_name]
                      for condition in self._edges[hierarchy_state.state_name][to_state]
                      if condition.is_call() and condition.is_context_satisfied(observation)]

        for call_edge in call_edges:
            called_automaton_condition = call_edge.call_condition

            new_context = called_automaton_condition.get_context().logic_and(hierarchy_state.context)
            new_stack = hierarchy_state.stack + [CallStackItem(hierarchy_state.state_name, call_edge.to_state_name,
                                                               self._name, called_automaton_condition,
                                                               hierarchy_state.context)]
            automaton = hierarchy.get_automaton(called_automaton_condition.get_called_automaton())
            initial_hierarchy_state = automaton.get_initial_hierarchy_state(new_context, new_stack)
            next_hierarchy_state = automaton.get_next_hierarchy_state_helper(initial_hierarchy_state, observation, hierarchy)
            if next_hierarchy_state is not None:
                next_hierarchy_states.append(next_hierarchy_state)

    def get_hierarchy_subgoals(self, subgoals: Set[FormulaCondition], context: DNFFormula, hierarchy, ignore_rejecting_transitions, only_satisfiable):
        for from_state in self._states:
            for to_state in self._edges[from_state]:
                self._get_hierarchy_subgoals_helper(subgoals, context, hierarchy, from_state, to_state, ignore_rejecting_transitions, only_satisfiable)

    def _get_hierarchy_subgoals_helper(self, subgoals: Set[FormulaCondition], context: DNFFormula, hierarchy, from_state,
                                       to_state, ignore_rejecting_transitions, only_satisfiable):
        for condition in self._edges[from_state][to_state]:
            if condition.is_call():
                called_automaton = hierarchy.get_automaton(condition.get_called_automaton())
                if self.is_initial_state(from_state):
                    # form a new context using the accumulated one and the context on the edge
                    called_automaton.get_hierarchy_subgoals(subgoals, context.logic_and(condition.get_context()), hierarchy,
                                                            ignore_rejecting_transitions, only_satisfiable)

                    # if the initial state can be reached from somewhere in the automaton, the call context is lost
                    if self.has_incoming_transitions(from_state):
                        called_automaton.get_hierarchy_subgoals(subgoals, condition.get_context(), hierarchy,
                                                                ignore_rejecting_transitions, only_satisfiable)
                else:
                    called_automaton.get_hierarchy_subgoals(subgoals, condition.get_context(), hierarchy,
                                                            ignore_rejecting_transitions, only_satisfiable)
            else:
                if not ignore_rejecting_transitions or not self.is_reject_state(to_state):
                    if self.is_initial_state(from_state):
                        # form a formula using the accumulated context and the formula in the edge
                        self._add_formula_disjuncts_to_subgoal_set(context.logic_and(condition.get_formula()), subgoals,
                                                                   only_satisfiable)

                        # if the initial state can be reached from somewhere in the automaton, the call context is lost
                        if self.has_incoming_transitions(from_state):
                            self._add_formula_disjuncts_to_subgoal_set(condition.get_formula(), subgoals, only_satisfiable)
                    else:
                        # for any state different from the initial state
                        self._add_formula_disjuncts_to_subgoal_set(condition.get_formula(), subgoals, only_satisfiable)

    def _add_formula_disjuncts_to_subgoal_set(self, formula: DNFFormula, subgoals: Set[EdgeCondition], only_satisfiable):
        """Takes the disjuncts in the DNF (atoms, negations or ands) and add them to the set."""
        for dnf_disjunct in formula.get_dnf_disjuncts(only_satisfiable):
            subgoals.add(FormulaCondition(dnf_disjunct))

    def get_num_automaton_subgoals(self):
        return len(self.get_automaton_subgoals())

    def get_automaton_subgoals(self) -> List[EdgeCondition]:
        """
        Returns a sorted list of the unique subgoals in the automaton. Unlike the analogous 'get_hierarchy_subgoals' method, this
        method operates in this automaton instance, and does not recursively explore the rest of the hierarchy forming
        global subgoal formulas. This method is used in the function approximation approach: a single metacontroller is
        kept for each automaton, and provides Q-value for each of the formulas in the automaton given the state of the
        environment, the accumulated context and an automaton state.
        """
        subgoals = set([dec_condition
                        for from_state in self._states
                        for to_state in self._edges[from_state]
                        for condition in self._edges[from_state][to_state]
                        for dec_condition in condition.decompose()])
        return sorted(subgoals)

    def get_automaton_subgoals_sat_mask(self, automaton_state, context, observations, hierarchy):
        sat_conditions = self.get_outgoing_conditions_with_terminating_paths(
            automaton_state, context, observations, hierarchy, False
        )
        if len(sat_conditions) > 0:
            sat_conditions, _ = zip(*sat_conditions)
        return [1 if c in sat_conditions else 0 for c in self.get_automaton_subgoals()]

    def _add_call_disjuncts_to_subgoal_set(self, condition: CallCondition, subgoals: Set[EdgeCondition]):
        """
        Takes the disjuncts from the context in a call condition and creates a new call condition for each of these
        disjuncts, which are then added to the set of subgoals. Note that the context of the new call condition is still
        a DNF formula but with only one disjunct.
        """
        for dec_condition in condition.decompose():
            subgoals.add(dec_condition)

    def plot(self, plot_folder, filename):
        """
        Plots the DFA into a file.

        Keyword arguments:
            plot_folder -- folder where the plot will be written.
            filename -- name of the file containing the plot.
            use_subprocess -- if True, it runs Graphviz from the command line; else, it runs graphviz from the Python
                              package.
        """
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        solution_plot_path = os.path.join(plot_folder, filename)
        diagram_path = os.path.join(plot_folder, HierarchicalAutomaton.GRAPHVIZ_SUBPROCESS_TXT)
        self._write_graphviz_diagram(diagram_path)
        self._run_graphviz_subprocess(diagram_path, solution_plot_path)

    def _write_graphviz_diagram(self, diagram_path):
        with open(diagram_path, 'w') as f:
            f.write(self._get_graphviz_diagram())

    def _get_graphviz_diagram(self):
        """Exports the automaton into a file using Graphviz format."""
        graphviz_str = "digraph G {\n"

        # write states
        for state in self._states:
            graphviz_str += f"{state} [label=\"{state}\"];\n"

        # write edges - collapsed edges means compressing OR conditions into single edges labelled with an OR in the
        #               middle
        for from_state in self._states:
            for to_state in self._edges[from_state]:
                for condition in self._edges[from_state][to_state]:
                    graphviz_str += f"{from_state} -> {to_state} [label=\"{condition}\"];\n"

        graphviz_str += "}"
        return graphviz_str

    def _run_graphviz_subprocess(self, diagram_path, filename):
        """
        Runs Graphviz from the command line and exports the automaton in .png format.

        Keyword arguments:
            diagram_path -- path to the .txt file containing the automaton in Graphviz format.
            filename -- output file name.
        """
        subprocess.call(["dot", "-Tpng", diagram_path, "-o", filename])
