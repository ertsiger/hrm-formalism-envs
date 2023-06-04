import gym
from gym_hierarchical_subgoal_automata.automata.common import CallStackItem, HierarchyState
from gym_hierarchical_subgoal_automata.automata.condition import CallCondition, FormulaCondition
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula, TRUE
from gym_hierarchical_subgoal_automata.envs.craftworld.craftworld_env import CraftWorldObservables
import unittest


class BaseTest(unittest.TestCase):
    def _test_trace(self, hierarchy, expected_hierarchy_states, trace):
        hierarchy_state = hierarchy.get_initial_state()
        self.assertEqual(expected_hierarchy_states[0], hierarchy_state)
        for t in range(1, len(trace) + 1):
            hierarchy_state = hierarchy.get_next_hierarchy_state(hierarchy_state, trace[t - 1])
            self.assertEqual(expected_hierarchy_states[t], hierarchy_state)

    def _test_subgoals(self, hierarchy, expected_subgoals):
        subgoals = set()
        hierarchy.get_subgoals(subgoals)
        self.assertEqual(expected_subgoals, subgoals)

    def _test_not_rejecting_subgoals(self, hierarchy, expected_subgoals):
        subgoals = set()
        hierarchy.get_subgoals(subgoals, True)
        self.assertEqual(expected_subgoals, subgoals)

    def _test_local_subgoals(self, automaton, expected_subgoals):
        self.assertEqual(expected_subgoals, automaton.get_automaton_subgoals())

    def _test_outgoing_conditions(self, automaton, state, expected_conditions):
        self.assertEqual(sorted(expected_conditions), sorted(automaton.get_outgoing_conditions(state)))


class BucketRejectingTest(BaseTest):
    def get_env(self):
        return gym.make("gym_hierarchical_subgoal_automata:CraftWorldBucket-v0", params={
            "environment_seed": 0, "include_deadends": True, "grid_params": {
                "grid_type": "open_plan", "width": 10, "height": 10, "use_lava": True, "num_lava": 1
            }
        })

    def test_all_subgoals(self):
        self._test_subgoals(
            self.get_env().get_hierarchy(),
            {
                FormulaCondition(DNFFormula([[CraftWorldObservables.IRON, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.TABLE, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.LAVA]]))
            }
        )

    def test_non_rej_subgoals(self):
        self._test_not_rejecting_subgoals(
            self.get_env().get_hierarchy(),
            {
                FormulaCondition(DNFFormula([[CraftWorldObservables.IRON, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.TABLE, f"~{CraftWorldObservables.LAVA}"]]))
            }
        )


class BookRejectingTest(BaseTest):
    def get_env(self):
        return gym.make("gym_hierarchical_subgoal_automata:CraftWorldBook-v0", params={
            "environment_seed": 0, "include_deadends": True, "grid_params": {
                "grid_type": "open_plan", "width": 10, "height": 10, "use_lava": True, "num_lava": 1
            }
        })

    def test_all_subgoals(self):
        self._test_subgoals(
            self.get_env().get_hierarchy(),
            {
                FormulaCondition(DNFFormula([[CraftWorldObservables.SUGARCANE, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.RABBIT, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.WORKBENCH, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.TABLE, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.LAVA]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.RABBIT, f"~{CraftWorldObservables.LAVA}", f"~{CraftWorldObservables.SUGARCANE}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.LAVA, f"~{CraftWorldObservables.LAVA}", f"~{CraftWorldObservables.SUGARCANE}"]]))
            }
        )

    def test_non_rej_subgoals(self):
        self._test_not_rejecting_subgoals(
            self.get_env().get_hierarchy(),
            {
                FormulaCondition(DNFFormula([[CraftWorldObservables.SUGARCANE, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.RABBIT, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.WORKBENCH, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.TABLE, f"~{CraftWorldObservables.LAVA}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.RABBIT, f"~{CraftWorldObservables.LAVA}", f"~{CraftWorldObservables.SUGARCANE}"]]))
            }
        )


class DisjunctionTest(BaseTest):
    def get_env(self):
        return gym.make("gym_hierarchical_subgoal_automata:CraftWorldTestDisjunction-v0", params={
            "environment_seed": 0, "grid_params": {"grid_type": "open_plan", "width": 10, "height": 10}
        })

    def test_iron_sugarcane_disjunction(self):
        """
        Tests the case in which the iron (i) and the sugarcane (s) are observed jointly (i.e., both symbols in the
        transition i|s in the root automaton are observed). The full context i|s is then propagated through the stack.
        """
        self._test_trace(
            self.get_env().get_hierarchy(),
            [
                HierarchyState("u0", "m0", TRUE, [], []),
                HierarchyState("u1", "m3", TRUE, [
                    CallStackItem("u0", "u_acc", "m0", CallCondition("m1", DNFFormula([[CraftWorldObservables.IRON], [CraftWorldObservables.SUGARCANE]])), TRUE),
                    CallStackItem("u0", "u1", "m1", CallCondition("m2", DNFFormula([[CraftWorldObservables.COW]])), DNFFormula([[CraftWorldObservables.IRON], [CraftWorldObservables.SUGARCANE]])),
                    CallStackItem("u0", "u_acc", "m2", CallCondition("m3", DNFFormula([[CraftWorldObservables.WHEAT]])), DNFFormula([[CraftWorldObservables.IRON, CraftWorldObservables.COW], [CraftWorldObservables.SUGARCANE, CraftWorldObservables.COW]]))
                ], []),
                HierarchyState("u1", "m1", TRUE, [
                    CallStackItem("u0", "u_acc", "m0", CallCondition("m1", DNFFormula([[CraftWorldObservables.IRON], [CraftWorldObservables.SUGARCANE]])), TRUE)], []),
                HierarchyState("u_acc", "m0", TRUE, [], [])
            ],
            [
                {CraftWorldObservables.IRON, CraftWorldObservables.SUGARCANE, CraftWorldObservables.COW, CraftWorldObservables.WHEAT, CraftWorldObservables.TABLE},
                {CraftWorldObservables.REDSTONE},
                {CraftWorldObservables.RABBIT}
            ]
        )

    def test_iron_disjunction(self):
        """
        Tests the case in which the iron (i) is observed alone (without the sugar), i.e. it is symbol responsible for
        satisfying the transition i|s in the root automaton. Therefore, only the i part of the context i|s is propagated
        through the stack.
        """
        self._test_trace(
            self.get_env().get_hierarchy(),
            [
                HierarchyState("u0", "m0", TRUE, [], []),
                HierarchyState("u1", "m3", TRUE, [
                    CallStackItem("u0", "u_acc", "m0", CallCondition("m1", DNFFormula([[CraftWorldObservables.IRON]])), TRUE),
                    CallStackItem("u0", "u1", "m1", CallCondition("m2", DNFFormula([[CraftWorldObservables.COW]])), DNFFormula([[CraftWorldObservables.IRON]])),
                    CallStackItem("u0", "u_acc", "m2", CallCondition("m3", DNFFormula([[CraftWorldObservables.WHEAT]])), DNFFormula([[CraftWorldObservables.IRON, CraftWorldObservables.COW]]))
                ], []),
                HierarchyState("u1", "m1", TRUE, [
                    CallStackItem("u0", "u_acc", "m0", CallCondition("m1", DNFFormula([[CraftWorldObservables.IRON]])), TRUE)], []),
                HierarchyState("u_acc", "m0", TRUE, [], [])
            ],
            [
                {CraftWorldObservables.IRON, CraftWorldObservables.COW, CraftWorldObservables.WHEAT, CraftWorldObservables.TABLE},
                {CraftWorldObservables.REDSTONE},
                {CraftWorldObservables.RABBIT}
            ]
        )

    def test_sugarcane_disjunction(self):
        """
        Tests the case in which the sugarcane (s) is observed alone (without the iron), i.e. it is symbol responsible for
        satisfying the transition i|s in the root automaton. Therefore, only the s part of the context i|s is propagated
        through the stack.
        """
        self._test_trace(
            self.get_env().get_hierarchy(),
            [
                HierarchyState("u0", "m0", TRUE, [], []),
                HierarchyState("u1", "m3", TRUE, [
                    CallStackItem("u0", "u_acc", "m0", CallCondition("m1", DNFFormula([[CraftWorldObservables.SUGARCANE]])), TRUE),
                    CallStackItem("u0", "u1", "m1", CallCondition("m2", DNFFormula([[CraftWorldObservables.COW]])), DNFFormula([[CraftWorldObservables.SUGARCANE]])),
                    CallStackItem("u0", "u_acc", "m2", CallCondition("m3", DNFFormula([[CraftWorldObservables.WHEAT]])), DNFFormula([[CraftWorldObservables.SUGARCANE, CraftWorldObservables.COW]]))
                ], []),
                HierarchyState("u1", "m1", TRUE, [
                    CallStackItem("u0", "u_acc", "m0", CallCondition("m1", DNFFormula([[CraftWorldObservables.SUGARCANE]])), TRUE)], []),
                HierarchyState("u_acc", "m0", TRUE, [], [])
            ],
            [
                {CraftWorldObservables.SUGARCANE, CraftWorldObservables.COW, CraftWorldObservables.WHEAT, CraftWorldObservables.TABLE},
                {CraftWorldObservables.REDSTONE},
                {CraftWorldObservables.RABBIT}
            ]
        )

    def test_hierarchy_subgoals(self):
        self._test_subgoals(
            self.get_env().get_hierarchy(), {
                FormulaCondition(DNFFormula([[CraftWorldObservables.SUGARCANE, CraftWorldObservables.COW,
                                              CraftWorldObservables.WHEAT, CraftWorldObservables.TABLE]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.IRON, CraftWorldObservables.COW,
                                              CraftWorldObservables.WHEAT, CraftWorldObservables.TABLE]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.REDSTONE]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.TABLE]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.SQUID, f"~{CraftWorldObservables.TABLE}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.SQUID, f"~{CraftWorldObservables.RABBIT}"]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.RABBIT]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.COW, CraftWorldObservables.WHEAT,
                                              CraftWorldObservables.TABLE]])),
                FormulaCondition(DNFFormula([[CraftWorldObservables.CHICKEN, f"~{CraftWorldObservables.REDSTONE}"]]))
            }
        )

    def test_root_subgoals(self):
        self._test_local_subgoals(
            self.get_env().get_hierarchy().get_root_automaton(),
            sorted([
                FormulaCondition(DNFFormula([[CraftWorldObservables.SQUID, f"~{CraftWorldObservables.TABLE}"]])),
                CallCondition("m1", DNFFormula([[CraftWorldObservables.IRON]])),
                CallCondition("m1", DNFFormula([[CraftWorldObservables.SUGARCANE]]))
            ])
        )

    def test_root_outgoing_conditions(self):
        automaton = self.get_env().get_hierarchy().get_root_automaton()
        self._test_outgoing_conditions(
            automaton,
            automaton.get_initial_state(),
            [
                (FormulaCondition(DNFFormula([[CraftWorldObservables.SQUID, f"~{CraftWorldObservables.TABLE}"]])), automaton.get_accept_state()),
                (CallCondition("m1", DNFFormula([[CraftWorldObservables.IRON]])), automaton.get_accept_state()),
                (CallCondition("m1", DNFFormula([[CraftWorldObservables.SUGARCANE]])), automaton.get_accept_state())
            ]
        )


class DisjunctionSimpleTest(BaseTest):
    def get_env(self):
        return gym.make("gym_hierarchical_subgoal_automata:CraftWorldTestDisjunctionSimple-v0", params={
            "environment_seed": 0, "grid_params": {"grid_type": "open_plan", "width": 10, "height": 10}
        })

    def test_hierarchy_subgoals(self):
        self._test_subgoals(
            self.get_env().get_hierarchy(),
            {FormulaCondition(DNFFormula([[CraftWorldObservables.IRON]])),
             FormulaCondition(DNFFormula([[CraftWorldObservables.SUGARCANE]]))}
        )


class MatchingLiteralsTest(unittest.TestCase):
    def test_all_matching(self):
        f1 = FormulaCondition(DNFFormula([["a", "~b", "~c"]]))
        f2 = FormulaCondition(DNFFormula([["a", "~b"]]))
        self.assertEqual(2, f1.get_num_matching_literals(f2))

    def test_pos_matching(self):
        f1 = FormulaCondition(DNFFormula([["a", "~b", "~c"]]))
        f2 = FormulaCondition(DNFFormula([["a", "~b"]]))
        self.assertEqual(1, f1.get_num_matching_pos_literals(f2))

    def test_all_matching_duplicated(self):
        f1 = FormulaCondition(DNFFormula([["a", "~b", "~c", "a", "~b", "~c"]]))
        f2 = FormulaCondition(DNFFormula([["a", "~b", "a", "~b"]]))
        self.assertEqual(2, f1.get_num_matching_literals(f2))

    def test_pos_matching_duplicated(self):
        f1 = FormulaCondition(DNFFormula([["a", "~b", "~c", "a", "~b", "~c"]]))
        f2 = FormulaCondition(DNFFormula([["a", "~b", "a", "~b"]]))
        self.assertEqual(1, f1.get_num_matching_pos_literals(f2))


if __name__ == "__main__":
    unittest.main()
