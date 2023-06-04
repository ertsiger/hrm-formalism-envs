from gym_hierarchical_subgoal_automata.automata.condition import CallCondition
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula
from typing import List, NamedTuple


class CallStackItem(NamedTuple):
    from_state_name: str
    to_state_name: str
    automaton_name: str
    call_condition: CallCondition
    context: DNFFormula


class SatisfiedCallContext(NamedTuple):
    call_condition: CallCondition
    to_state_name: str


class SatisfiedCall(NamedTuple):
    automaton_name: str
    call_condition: CallCondition


class HierarchyState(NamedTuple):
    state_name: str
    automaton_name: str
    context: DNFFormula
    stack: List[CallStackItem]
    satisfied_calls: List[SatisfiedCall]

    def get_stack_len(self):
        return len(self.stack)

    def is_stack_empty(self):
        return self.get_stack_len() == 0

    def get_stack_top(self):
        if self.get_stack_len() > 0:
            return self.stack[-1]
        raise RuntimeError("Error: The top element of an empty stack cannot be recovered.")

    def get_substack(self):
        if self.get_stack_len() > 0:
            return self.stack[:-1]
        raise RuntimeError("Error: The substack of an empty stack cannot be recovered.")

    def __eq__(self, other):
        return self.state_name == other.state_name and \
               self.automaton_name == other.automaton_name and \
               self.context == other.context and \
               self.stack == other.stack

    def __ne__(self, other):
        return not (self == other)


def get_param(params, param_name, default_value=None):
    if params is not None and param_name in params:
        return params[param_name]
    return default_value
