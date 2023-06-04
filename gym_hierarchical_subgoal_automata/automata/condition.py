from abc import ABC, abstractmethod
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula
from typing import Set


class EdgeCondition(ABC):
    @abstractmethod
    def is_call(self):
        pass

    @abstractmethod
    def get_called_automaton(self):
        pass

    @abstractmethod
    def get_context(self):
        pass

    @abstractmethod
    def get_formula(self):
        pass

    @abstractmethod
    def decompose(self):
        pass

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return str(self)


class CallCondition(EdgeCondition):
    def __init__(self, called_automaton: str, context: DNFFormula):
        super(CallCondition, self).__init__()
        self._called_automaton = called_automaton
        self._context = context

    def is_call(self):
        return True

    def get_called_automaton(self):
        return self._called_automaton

    def get_context(self):
        return self._context

    def get_formula(self):
        raise NotImplementedError(f"Error: The get_formula() method is not available for class "
                                  f"{CallCondition.__name__}.")

    def decompose(self):
        """
        Decomposes the CallCondition into (potentially) several CallConditions: one for each disjunct in the DNFFormula
        representing the context.
        """
        for disjunct_dnf in self._context.get_dnf_disjuncts():
            yield CallCondition(self._called_automaton, disjunct_dnf)

    def is_context_satisfied(self, observation: Set[str]):
        return self._context.is_satisfied(observation)

    def get_satisfied_condition(self, observation: Set[str]):
        return CallCondition(self._called_automaton, self._context.get_satisfied_dnf(observation))

    def __eq__(self, other):
        if not other.is_call():
            return False
        return self._called_automaton == other.get_called_automaton() and self._context == other.get_context()

    def __lt__(self, other):
        if not other.is_call():
            return False  # formula conditions precede call conditions
        if self._called_automaton == other.get_called_automaton():
            return self._context < other.get_context()
        return self._called_automaton < other.get_called_automaton()

    def __hash__(self):
        return hash((self._called_automaton, self._context))

    def __str__(self):
        return f"{self._context}/{self._called_automaton}"


class FormulaCondition(EdgeCondition):
    def __init__(self, formula: DNFFormula):
        assert isinstance(formula, DNFFormula)
        super(FormulaCondition, self).__init__()
        self._formula = formula

    def is_call(self):
        return False

    def get_called_automaton(self):
        raise NotImplementedError(f"Error: The get_called_automaton() method is not available for class "
                                  f"{FormulaCondition.__name__}.")

    def get_context(self):
        raise NotImplementedError(f"Error: The get_context() method is not available for class "
                                  f"{FormulaCondition.__name__}.")

    def get_formula(self):
        return self._formula

    def decompose(self):
        """
        Decomposes a FormulaCondition into (potentially) several FormulaConditions if the contained formula is a DNF
        which is not TRUE.
        """
        for dnf_disjunct in self._formula.get_dnf_disjuncts():
            yield FormulaCondition(dnf_disjunct)

    def is_satisfied(self, observation: Set[str]):
        return self._formula.is_satisfied(observation)

    def get_num_matching_literals(self, other):
        assert not other.is_call()
        if len(self._formula) == 1 and len(other.get_formula()) == 1:
            return self._formula.get_formula()[0].get_num_matching_literals(other.get_formula().get_formula()[0])
        raise RuntimeError("Error: This method is applicable only to the case in which the DNF formulas contain one"
                           "disjunct.")

    def get_num_matching_pos_literals(self, other):
        assert not other.is_call()
        if len(self._formula) == 1 and len(other.get_formula()) == 1:
            return self._formula.get_formula()[0].get_num_matching_pos_literals(other.get_formula().get_formula()[0])
        raise RuntimeError("Error: This method is applicable only to the case in which the DNF formulas contain one"
                           "disjunct.")

    def __eq__(self, other):
        if other.is_call():
            return False
        return self._formula == other.get_formula()

    def __lt__(self, other):
        if other.is_call():
            return True  # formula conditions precede call conditions
        return self._formula < other.get_formula()

    def __hash__(self):
        return hash(self._formula)

    def __str__(self):
        return str(self._formula)
