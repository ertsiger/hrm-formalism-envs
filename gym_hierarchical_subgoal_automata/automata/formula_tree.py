from gym_hierarchical_subgoal_automata.automata.condition import FormulaCondition
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula, TRUE
from typing import Dict, List, Optional, Set, Tuple


def get_conjunction_formula(formula_condition: FormulaCondition):
    dnf_formula = formula_condition.get_formula().get_formula()
    assert len(dnf_formula) == 1  # all the formulas in the bank are DNFs with a single disjunct/conjunction
    return dnf_formula[0]


class FormulaNode:
    def __init__(self, formula: FormulaCondition, children=None, parent=None):
        self.formula = formula
        self.children: List[FormulaNode] = children if children is not None else []
        self.parent: Optional[FormulaNode] = parent

    def get_formula_condition(self):
        return self.formula

    def subsumes(self, other):
        if self.formula.get_formula().is_true():
            return True
        if other.get_formula_condition().get_formula().is_true():
            return False
        cf1, cf2 = get_conjunction_formula(self.formula), get_conjunction_formula(other.get_formula_condition())
        return cf1.subsumes(cf2)

    def is_subsumed(self, other):
        if other.get_formula_condition().get_formula().is_true():
            return True
        if self.formula.get_formula().is_true():
            return False
        cf1, cf2 = get_conjunction_formula(self.formula), get_conjunction_formula(other.get_formula_condition())
        return cf2.subsumes(cf1)

    def get_subsumed_children(self, other, observations: Set[Tuple[str]]):
        return [x for x in self.children if x.is_subsumed(other) and x.has_equal_observation_covering(other, observations)]

    def find_subsuming_child(self, other, observations: Set[Tuple[str]]):
        """
        Returns a child node that subsumes the formula passed as a parameter. The child node must also have an equal
        covering of the passed observations. There might be multiple children nodes that subsume the formula but only
        one is returned. For example, if the node is 'a' and has to children 'a&~b' and 'a&~c', the formula (other)
        'a&~b&~c' could potentially be a children of either of these two. However, it does not really matter which one
        of them is: if we find an inconsistency later on, we will amend it.
        """
        for child in self.children:
            if child.subsumes(other) and child.has_equal_observation_covering(other, observations):
                return child
        return None

    def has_equal_observation_covering(self, other, observations: Set[Tuple[str]]):
        """
        Returns True if the node's formula and the passed formula have the same truth values for all observations passed
        as parameters. Else, it returns False.
        """
        for obs in observations:
            obs_set = set(obs)
            if self.is_satisfied(obs_set) ^ other.is_satisfied(obs_set):
                return False
        return True

    def is_satisfied(self, observation: Set[str]):
        return self.formula.is_satisfied(observation)

    def add_child(self, node):
        self.children.append(node)

    def add_children(self, nodes):
        self.children.extend(nodes)

    def remove_child(self, node):
        self.children.remove(node)

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def is_abs_root(self):
        return self.parent is None

    def is_root(self):
        if self.is_abs_root():
            return False
        return self.parent.is_abs_root()

    def __str__(self):
        return str(self.formula)


class FormulaTree:
    def __init__(self):
        self.root_node = FormulaNode(FormulaCondition(TRUE))
        self.formula_nodes: Dict[FormulaCondition, FormulaNode] = {}
        self.observations: Set[Tuple[str]] = set()

    def add_formula(self, formula: FormulaCondition):
        # We assume that all formulas in our bank have at least one positive literal
        assert not self.contains(formula) and self._is_valid_formula(formula)

        formula_node = FormulaNode(formula)
        self.formula_nodes[formula] = formula_node
        self._insert_formula_node(formula_node)

    def get_observations(self):
        return self.observations

    def _is_valid_formula(self, formula_condition: FormulaCondition):
        return len(get_conjunction_formula(formula_condition).get_pos_literals()) > 0

    def _insert_formula_node(self, formula_node: FormulaNode):
        # Start from root level: check if formula subsumes
        current_node = self.root_node
        added_node = False

        while not added_node:
            # Find a child of the current node whose formula subsumes the new one.
            child_node = current_node.find_subsuming_child(formula_node, self.observations)

            if child_node is not None:
                # If we find a node that subsumes the formula, then we keep exploring down this path in the tree.
                current_node = child_node
            else:
                # If no formula subsumes the current one, check if the current one subsumes any of those formulas.
                # If (some of the) children of the current node are subsumed by the formula, then it means that the
                # formula should be its parent (i.e., in the middle between current_node and the children). Thus, we
                # add these children to our new node and remove them from the old one.
                subsumed_children = current_node.get_subsumed_children(formula_node, self.observations)
                for child in subsumed_children:
                    current_node.remove_child(child)
                    formula_node.add_child(child)
                    child.set_parent(formula_node)

                # The new formula node becomes a child of the current node.
                current_node.add_child(formula_node)
                formula_node.set_parent(current_node)
                added_node = True

    def contains(self, formula: FormulaCondition):
        return formula in self.formula_nodes

    def on_observation(self, observation: Set[str]):
        """
        Adds the observation of the set of observations if it hasn't been seen before and repairs inconsistencies in the
        formula tree. Returns True if the observation is new, and False otherwise.
        """
        if not self._was_observed(observation):
            self.observations.add(self._tuplify_obs(observation))
            self._repair_inconsistencies(observation)
            return True
        return False

    def _was_observed(self, observation: Set[str]):
        return self._tuplify_obs(observation) in self.observations

    def _tuplify_obs(self, observation: Set[str]):
        return tuple(sorted(observation))

    def _repair_inconsistencies(self, observation: Set[str]):
        inconsistent_nodes = []
        self._find_bank_inconsistent_nodes(observation, inconsistent_nodes)
        self._reinsert_inconsistent_nodes(inconsistent_nodes)

    def _find_bank_inconsistent_nodes(self, observation: Set[str], inconsistent_nodes: List[FormulaNode]):
        for child_node in self.root_node.children:
            self._find_inconsistent_nodes_helper(observation, child_node, inconsistent_nodes)

    def _find_inconsistent_nodes_helper(self, observation: Set[str], node: FormulaNode, inconsistent_nodes: List[FormulaNode]):
        for child_node in node.children:
            if node.is_satisfied(observation) ^ child_node.is_satisfied(observation):
                inconsistent_nodes.append(child_node)
            else:
                self._find_inconsistent_nodes_helper(observation, child_node, inconsistent_nodes)

    def _reinsert_inconsistent_nodes(self, inconsistent_nodes: List[FormulaNode]):
        for node in inconsistent_nodes:
            # Remove child from parent and nullify parenting
            parent_node = node.get_parent()
            parent_node.remove_child(node)
            node.set_parent(None)

            # Reinsert the node in the forest
            self._insert_formula_node(node)

    def get_root(self, formula: FormulaCondition):
        node = self.formula_nodes[formula]
        if node.is_abs_root():
            return node  # this case should not occur, we are interested in children of the absolute
        current_node = node
        while not current_node.get_parent().is_abs_root():
            current_node = current_node.get_parent()
        return current_node

    def get_root_formula(self, formula: FormulaCondition):
        return self.get_root(formula).get_formula_condition()

    def _is_formula_condition_sat_by_seen_obs(self, formula_condition: FormulaCondition):
        for obs in self.observations:
            if formula_condition.is_satisfied(set(obs)):
                return True
        return False

    def __str__(self):
        bank_str = ""
        for node in self.root_node.children:
            bank_str += self._str_node(node, 1)
        return bank_str

    def _str_node(self, node, level):
        bank_str = f"{'-' * level}{node}\n"
        for child in node.children:
            bank_str += self._str_node(child, level + 2)
        return bank_str


if __name__ == "__main__":
    tree = FormulaTree()

    tree.add_formula(FormulaCondition(DNFFormula([["a"]])))
    tree.add_formula(FormulaCondition(DNFFormula([["a", "~b"]])))
    tree.add_formula(FormulaCondition(DNFFormula([["a", "~c"]])))
    tree.add_formula(FormulaCondition(DNFFormula([["a", "~c", "~d"]])))
    tree.add_formula(FormulaCondition(DNFFormula([["a", "~b", "~c"]])))
    tree.add_formula(FormulaCondition(DNFFormula([["a", "~b", "~c", "~d"]])))

    print(tree)
    print("Root:", tree.get_root(FormulaCondition(DNFFormula([["a", "~b", "~c", "~d"]]))))

    tree.on_observation({"a", "c"})
    print(tree)

    tree.on_observation({"a", "d"})
    print(tree)

    print("Root:", tree.get_root(FormulaCondition(DNFFormula([["a", "~b", "~c", "~d"]]))))
