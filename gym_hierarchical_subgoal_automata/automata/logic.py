from abc import ABC, abstractmethod
import numpy as np
from typing import List, Sequence, Set


class Formula(ABC):
    @abstractmethod
    def get_formula(self):
        pass

    @abstractmethod
    def is_satisfied(self, observation: Set[str]):
        """
        Returns true if a formula is satisfied by an observation. An observation is interpreted as a truth assignment
        where observables are true if they appear in the observation and false otherwise.
        """
        pass

    def is_negative_literal(self, literal):
        return literal.startswith('~')

    def get_symbol_from_literal(self, literal):
        if self.is_negative_literal(literal):
            return literal[1:]
        return literal

    @abstractmethod
    def get_embedding(self, observables: List[str]):
        pass

    @abstractmethod
    def to_dnf(self):
        pass

    @abstractmethod
    def is_satisfiable(self):
        pass

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.get_formula())


class ConjunctionFormula(Formula):
    def __init__(self, formula: Sequence[str]):
        # The set removes duplicate literals, sorted makes sure that the formula always looks the same, and tuple makes
        # sure we can use it in mappings.
        self._formula = tuple(sorted(set(formula)))

    def get_formula(self):
        return self._formula

    def is_satisfied(self, observation: Set[str]):
        for literal in self._formula:
            is_neg = self.is_negative_literal(literal)
            symbol = self.get_symbol_from_literal(literal)
            if (is_neg and symbol in observation) or (not is_neg and symbol not in observation):
                return False
        return True

    def get_embedding(self, observables: List[str]):
        formula_v = np.zeros(len(observables))
        for literal in self._formula:
            idx = observables.index(self.get_symbol_from_literal(literal))
            if self.is_negative_literal(literal):
                formula_v[idx] = -1
            else:
                formula_v[idx] = 1
        return formula_v

    def get_num_matching_literals(self, other):
        assert isinstance(other, ConjunctionFormula)
        s1, s2 = set(self._formula), set(other.get_formula())
        return len(s1.intersection(s2))

    def get_num_matching_pos_literals(self, other):
        assert isinstance(other, ConjunctionFormula)
        return len(self.get_pos_literals().intersection(other.get_pos_literals()))

    def subsumes(self, other):
        """
        A conjunction X subsumes another conjunction Y if all literals of X appear in Y and there is some literal of Y
        that does not appear in X. Therefore, we can just check if the set of literals for X is a subset of Y's.
        """
        s1, s2 = set(self._formula), set(other.get_formula())
        return s1.issubset(s2)

    def get_pos_literals(self):
        return set([l for l in self._formula if not l.startswith("~")])

    def to_dnf(self):
        return DNFFormula([self._formula])

    def is_satisfiable(self):
        for i in range(len(self._formula) - 1):
            for j in range(i + 1, len(self._formula)):
                if self.get_symbol_from_literal(self._formula[i]) == self.get_symbol_from_literal(self._formula[j]) and \
                    self.is_negative_literal(self._formula[i]) ^ self.is_negative_literal(self._formula[j]):
                        return False
        return True

    def __str__(self):
        return '&'.join(self._formula)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        assert isinstance(other, ConjunctionFormula)
        return self._formula == other.get_formula()

    def __lt__(self, other):
        assert isinstance(other, ConjunctionFormula)
        return self._formula < other.get_formula()

    def __hash__(self):
        return hash(self._formula)


class DNFFormula(Formula):
    def __init__(self, formula: Sequence[Sequence[str]]):
        dnf_formula = [ConjunctionFormula(disjunct) for disjunct in formula]
        dnf_formula.sort()
        self._formula = tuple(dnf_formula)

    def get_formula(self):
        return self._formula

    def is_satisfied(self, observation: Set[str]):
        if self.is_true():
            return True
        for disjunct in self._formula:
            if disjunct.is_satisfied(observation):
                return True
        return False

    def get_satisfied_dnf(self, observation: Set[str]):
        if self.is_true():
            return self
        satisfied_dnf = []
        for disjunct in self._formula:
            if disjunct.is_satisfied(observation):
                satisfied_dnf.append(disjunct.get_formula())
        if len(satisfied_dnf) > 0:
            return DNFFormula(satisfied_dnf)
        return None

    def get_embedding(self, observables: List[str]):
        if len(self._formula) == 0:
            return np.zeros(len(observables))
        elif len(self._formula) == 1:
            return self._formula[0].get_embedding(observables)

        # To support formulas with multiple disjuncts, we will have to concatenate the embeddings for each disjunct.
        # However, it is not as simple as that: we will have to know which is the maximum embeddin size according to
        # the constraints we impose in the automaton learning: the maximum number of disjuncts in a formula AND the
        # maximum depth of the hierarchy. Note that the embedding size is not only bounded by the former since contexts
        # can be aggregated, leading to DNFs with many more disjuncts! Since the max. number of disjuncts is usually 1,
        # we make this assumption by now.
        raise RuntimeError("Error: Formulas with more than one disjunct cannot be currently returned as embeddings.")

    def to_dnf(self):
        return self

    def logic_and(self, other):
        assert isinstance(other, DNFFormula)

        if self.is_true():
            return other
        if other.is_true():
            return self

        dnf_formula = []
        for df1 in self._formula:
            for df2 in other.get_formula():
                dnf_formula.append([*df1.get_formula(), *df2.get_formula()])
        return DNFFormula(dnf_formula)

    def get_dnf_disjuncts(self, only_satisfiable=False):
        """
        Generates a DNF formula for each disjunct in this DNF formula.
        """
        if self.is_true():
            yield self
        else:
            for disjunct in self._formula:
                if not only_satisfiable or disjunct.is_satisfiable():
                    yield disjunct.to_dnf()

    def is_true(self):
        return len(self._formula) == 0

    def contains(self, other):
        if other.is_true():
            return self.is_true()

        if self.is_true():
            return other.is_true()

        for disjunct in other.get_formula():
            if disjunct not in self._formula:
                return False
        return True

    def is_satisfiable(self):
        if self.is_true():
            return True
        for disjunct in self._formula:
            if disjunct.is_satisfiable():
                return True
        return False

    def __str__(self):
        if self.is_true():
            return "T"
        elif len(self._formula) == 1:
            return str(self._formula[0])
        return '|'.join([f"({disjunct})" for disjunct in self._formula])

    def __eq__(self, other):
        assert isinstance(other, DNFFormula)
        return self._formula == other.get_formula()

    def __lt__(self, other):
        assert isinstance(other, DNFFormula)
        return self._formula < other.get_formula()

    def __hash__(self):
        return hash(self._formula)


TRUE = DNFFormula([])
