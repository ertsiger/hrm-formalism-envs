from abc import ABC
from gym import spaces
from gym_hierarchical_subgoal_automata.automata.common import get_param
from gym_hierarchical_subgoal_automata.automata.hierarchical_automaton import HierarchicalAutomaton
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula, TRUE
from gym_hierarchical_subgoal_automata.envs.base.base_env import BaseEnv, TaskEnum
from gym_hierarchical_subgoal_automata.envs.craftworld.minigrid import CustomMiniGrid
from gym_hierarchical_subgoal_automata.envs.craftworld.wrappers import FullyObsTransform, OneHotWrapper, TabularWrapper
from gym_minigrid.window import Window
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, ReseedWrapper


class CraftWorldObservables:
    CHICKEN = "z"
    COW = "c"
    IRON = "i"
    LAVA = "l"
    RABBIT = "y"
    REDSTONE = "r"
    SQUID = "x"
    SUGARCANE = "s"
    TABLE = "t"
    WHEAT = "w"
    WORKBENCH = "h"


class CraftWorldTasks(TaskEnum):
    # Basic tasks (Level 0) - Just used for debugging
    CHICKEN = "get-chicken"
    COW = "get-cow"
    IRON = "get-iron"
    LAVA = "get-lava"
    RABBIT = "get-rabbit"
    REDSTONE = "get-redstone"
    SQUID = "get-squid"
    SUGARCANE = "get-sugarcane"
    TABLE = "get-table"
    WHEAT = "get-wheat"
    WORKBENCH = "get-workbench"

    # Level 1 tasks
    BATTER = "batter"
    BUCKET = "bucket"
    COMPASS = "compass"
    LEATHER = "leather"
    PAPER = "paper"
    QUILL = "quill"
    SUGAR = "sugar"

    # Level 2 tasks
    BOOK = "book"
    MAP = "map"
    MILK_BUCKET = "milk-bucket"
    TEN_PAPERS = "ten-papers"

    # Level 3 tasks
    BOOK_AND_QUILL = "book-and-quill"
    MILK_BUCKET_AND_SUGAR = "milk-bucket-and-sugar"

    # Level 4 tasks
    CAKE = "cake"

    # Other debugging tasks
    TEST_LOOP = "test-loop"
    TEST_CONTEXT = "test-context"
    TEST_DISJUNCTION = "test-disjunction"
    TEST_DISJUNCTION_SIMPLE = "test-disjunction-simple"
    TEST_DISJUNCTION_DOUBLE = "test-disjunction-double"
    TEST_SIMULTANEOUS_SAT = "test-simultaneous-sat"


OBJ_TO_OBSERVABLE = {
    "chicken": CraftWorldObservables.CHICKEN,
    "cow": CraftWorldObservables.COW,
    "iron": CraftWorldObservables.IRON,
    "lava": CraftWorldObservables.LAVA,
    "rabbit": CraftWorldObservables.RABBIT,
    "redstone": CraftWorldObservables.REDSTONE,
    "squid": CraftWorldObservables.SQUID,
    "sugarcane": CraftWorldObservables.SUGARCANE,
    "table": CraftWorldObservables.TABLE,
    "wheat": CraftWorldObservables.WHEAT,
    "workbench": CraftWorldObservables.WORKBENCH
}


class CraftWorldEnv(BaseEnv, ABC):
    STATE_FORMAT = "state_format"
    STATE_FORMAT_TABULAR = "tabular"
    STATE_FORMAT_ONE_HOT = "one_hot"
    STATE_FORMAT_FULL_OBS = "full_obs"

    SCALE_OBS_MODE = "scale_obs_mode"

    REMOVE_COLORS = "remove_colors"

    TASK_NAME_TO_AUTOMATON_NAME = {
        task_name: f"m{index}"
        for index, task_name in enumerate(CraftWorldTasks.list())
    }

    def __init__(self, task_name, params):
        super().__init__(params)

        self.task_name = task_name

        self.state_format = get_param(params, CraftWorldEnv.STATE_FORMAT, CraftWorldEnv.STATE_FORMAT_FULL_OBS)

        self.env = CustomMiniGrid(params, self.seed)
        self.include_deadends = self.env.contains_lava()
        if self.state_format == CraftWorldEnv.STATE_FORMAT_TABULAR:
            self.env = TabularWrapper(self.env)
        elif self.state_format == CraftWorldEnv.STATE_FORMAT_ONE_HOT:
            self.env = OneHotWrapper(self.env)
        elif self.state_format == CraftWorldEnv.STATE_FORMAT_FULL_OBS:
            self.env = FullyObsWrapper(self.env)
            self.env = ImgObsWrapper(self.env)
            self.env = FullyObsTransform(
                self.env,
                get_param(params, CraftWorldEnv.SCALE_OBS_MODE, "minus_one_one"),
                get_param(params, CraftWorldEnv.REMOVE_COLORS, False)
            )
        if not self.random_restart:
            self.env = ReseedWrapper(self.env, [self.seed])

        self.observation_space = self.env.observation_space
        self.action_space = spaces.Discrete(3)  # use the first 3 actions only (left, right, forward)
        self.num_directions = self.env.NUM_DIRECTIONS

        self.game_window = None

    def env_step(self, action):
        obs, _, _, info = self.env.step(action)
        return obs, info

    def get_observables(self):
        observables = [
            CraftWorldObservables.IRON, CraftWorldObservables.TABLE, CraftWorldObservables.COW,
            CraftWorldObservables.SUGARCANE, CraftWorldObservables.WHEAT, CraftWorldObservables.CHICKEN,
            CraftWorldObservables.REDSTONE, CraftWorldObservables.RABBIT, CraftWorldObservables.SQUID,
            CraftWorldObservables.WORKBENCH
        ]
        if self.include_deadends:
            observables.append(CraftWorldObservables.LAVA)
        return observables

    def get_restricted_observables(self):
        return self._get_restricted_observables_for_task(self.task_name)

    def _get_restricted_observables_for_task(self, task_name):
        if task_name.startswith("get-"):
            obj_name = task_name[len("get-"):]
            return self._get_deadend_extended_restricted_observables([OBJ_TO_OBSERVABLE[obj_name]])
        elif task_name == CraftWorldTasks.BATTER.value:
            return self._get_deadend_extended_restricted_observables([CraftWorldObservables.WHEAT, CraftWorldObservables.CHICKEN, CraftWorldObservables.TABLE])
        elif task_name == CraftWorldTasks.BUCKET.value:
            return self._get_deadend_extended_restricted_observables([CraftWorldObservables.IRON, CraftWorldObservables.TABLE])
        elif task_name == CraftWorldTasks.COMPASS.value:
            return self._get_deadend_extended_restricted_observables([CraftWorldObservables.IRON, CraftWorldObservables.REDSTONE, CraftWorldObservables.WORKBENCH])
        elif task_name == CraftWorldTasks.LEATHER.value:
            return self._get_deadend_extended_restricted_observables([CraftWorldObservables.RABBIT, CraftWorldObservables.WORKBENCH])
        elif task_name == CraftWorldTasks.PAPER.value:
            return self._get_deadend_extended_restricted_observables([CraftWorldObservables.SUGARCANE, CraftWorldObservables.WORKBENCH])
        elif task_name == CraftWorldTasks.QUILL.value:
            return self._get_deadend_extended_restricted_observables([CraftWorldObservables.CHICKEN, CraftWorldObservables.SQUID, CraftWorldObservables.TABLE])
        elif task_name == CraftWorldTasks.SUGAR.value:
            return self._get_deadend_extended_restricted_observables([CraftWorldObservables.SUGARCANE, CraftWorldObservables.TABLE])
        elif task_name == CraftWorldTasks.BOOK.value:
            return self._get_restricted_observables_from_dependencies([CraftWorldObservables.TABLE], [CraftWorldTasks.PAPER.value, CraftWorldTasks.LEATHER.value])
        elif task_name == CraftWorldTasks.MAP.value:
            return self._get_restricted_observables_from_dependencies([CraftWorldObservables.TABLE], [CraftWorldTasks.PAPER.value, CraftWorldTasks.COMPASS.value])
        elif task_name == CraftWorldTasks.MILK_BUCKET.value:
            return self._get_deadend_extended_restricted_observables([CraftWorldObservables.IRON, CraftWorldObservables.TABLE, CraftWorldObservables.COW])
        elif task_name == CraftWorldTasks.TEN_PAPERS.value:
            return self._get_restricted_observables_for_task(CraftWorldTasks.PAPER.value)
        elif task_name == CraftWorldTasks.BOOK_AND_QUILL.value:
            return self._get_restricted_observables_from_dependencies([], [CraftWorldTasks.BOOK.value, CraftWorldTasks.QUILL.value])
        elif task_name == CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value:
            return self._get_restricted_observables_from_dependencies([], [CraftWorldTasks.MILK_BUCKET.value, CraftWorldTasks.SUGAR.value])
        elif task_name == CraftWorldTasks.CAKE.value:
            return self._get_restricted_observables_from_dependencies([CraftWorldObservables.WORKBENCH], [CraftWorldTasks.BATTER.value, CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value])
        elif task_name == CraftWorldTasks.TEST_LOOP.value:
            return [CraftWorldObservables.IRON, CraftWorldObservables.TABLE, CraftWorldObservables.WHEAT,
                    CraftWorldObservables.SUGARCANE, CraftWorldObservables.CHICKEN]
        elif task_name == CraftWorldTasks.TEST_CONTEXT.value:
            return [CraftWorldObservables.IRON, CraftWorldObservables.TABLE, CraftWorldObservables.COW,
                    CraftWorldObservables.SUGARCANE, CraftWorldObservables.SQUID]
        elif task_name == CraftWorldTasks.TEST_DISJUNCTION.value:
            return [CraftWorldObservables.IRON, CraftWorldObservables.COW, CraftWorldObservables.WHEAT,
                    CraftWorldObservables.SUGARCANE, CraftWorldObservables.REDSTONE, CraftWorldObservables.TABLE,
                    CraftWorldObservables.RABBIT, CraftWorldObservables.SQUID, CraftWorldObservables.CHICKEN]
        elif task_name == CraftWorldTasks.TEST_DISJUNCTION_SIMPLE.value:
            return [CraftWorldObservables.IRON, CraftWorldObservables.SUGARCANE]
        elif task_name == CraftWorldTasks.TEST_DISJUNCTION_DOUBLE.value:
            return [CraftWorldObservables.IRON, CraftWorldObservables.TABLE, CraftWorldObservables.COW,
                    CraftWorldObservables.SUGARCANE, CraftWorldObservables.WHEAT, CraftWorldObservables.CHICKEN,
                    CraftWorldObservables.REDSTONE]
        elif task_name == CraftWorldTasks.TEST_SIMULTANEOUS_SAT:
            return [CraftWorldObservables.IRON, CraftWorldObservables.TABLE, CraftWorldObservables.COW,
                    CraftWorldObservables.SUGARCANE, CraftWorldObservables.SQUID]

    def _get_restricted_observables_from_dependencies(self, base_observables, dependencies):
        observables = base_observables
        for dependency in dependencies:
            observables.extend(self._get_restricted_observables_for_task(dependency))
        return list(set(self._get_deadend_extended_restricted_observables(observables)))

    def _get_deadend_extended_restricted_observables(self, observables):
        if self.include_deadends:
            observables.append(CraftWorldObservables.LAVA)
        return observables

    def get_observation(self):
        obj = self.env.get_current_object()
        observation = set()
        if obj is not None and obj.type in OBJ_TO_OBSERVABLE:
            observation.add(OBJ_TO_OBSERVABLE[obj.type])
        return observation

    def get_possible_observations(self):
        observations = [set()]
        for observable in [
            CraftWorldObservables.CHICKEN, CraftWorldObservables.COW, CraftWorldObservables.IRON,
            CraftWorldObservables.RABBIT, CraftWorldObservables.REDSTONE, CraftWorldObservables.SQUID,
            CraftWorldObservables.SUGARCANE, CraftWorldObservables.TABLE, CraftWorldObservables.WHEAT,
            CraftWorldObservables.WORKBENCH
        ]:
            observations.append(set(observable))
        if self.include_deadends:
            observations.append(set(CraftWorldObservables.LAVA))
        observations.sort(key=lambda x: tuple(x))
        return observations

    def get_hierarchy(self):
        return self._get_hierarchy_for_task(self.task_name)

    def _get_hierarchy_for_task(self, task_name):
        if task_name.startswith("get-"):
            return self._get_flat_one_subgoal_hierarchy(task_name)
        elif task_name == CraftWorldTasks.BATTER.value:
            return self._get_flat_diamond_hierarchy(task_name, CraftWorldObservables.WHEAT, CraftWorldObservables.CHICKEN, CraftWorldObservables.TABLE)
        elif task_name == CraftWorldTasks.BUCKET.value:
            return self._get_flat_two_subgoal_hierarchy(task_name, CraftWorldObservables.IRON, CraftWorldObservables.TABLE)
        elif task_name == CraftWorldTasks.COMPASS.value:
            return self._get_flat_diamond_hierarchy(task_name, CraftWorldObservables.IRON, CraftWorldObservables.REDSTONE, CraftWorldObservables.WORKBENCH)
        elif task_name == CraftWorldTasks.LEATHER.value:
            return self._get_flat_two_subgoal_hierarchy(task_name, CraftWorldObservables.RABBIT, CraftWorldObservables.WORKBENCH)
        elif task_name == CraftWorldTasks.PAPER.value:
            return self._get_flat_two_subgoal_hierarchy(task_name, CraftWorldObservables.SUGARCANE, CraftWorldObservables.WORKBENCH)
        elif task_name == CraftWorldTasks.QUILL.value:
            return self._get_flat_diamond_hierarchy(task_name, CraftWorldObservables.CHICKEN, CraftWorldObservables.SQUID, CraftWorldObservables.TABLE)
        elif task_name == CraftWorldTasks.SUGAR.value:
            return self._get_flat_two_subgoal_hierarchy(task_name, CraftWorldObservables.SUGARCANE, CraftWorldObservables.TABLE)
        elif task_name == CraftWorldTasks.BOOK.value:
            return self._get_book_hierarchy()
        elif task_name == CraftWorldTasks.MAP.value:
            return self._get_map_hierarchy()
        elif task_name == CraftWorldTasks.MILK_BUCKET.value:
            return self._get_milk_bucket_hierarchy()
        elif task_name == CraftWorldTasks.TEN_PAPERS.value:
            return self._get_ten_papers_hierarchy()
        elif task_name == CraftWorldTasks.BOOK_AND_QUILL.value:
            return self._get_book_and_quill_hierarchy()
        elif task_name == CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value:
            return self._get_milk_bucket_and_sugar_hierarchy()
        elif task_name == CraftWorldTasks.CAKE.value:
            return self._get_cake_hierarchy()
        elif task_name == CraftWorldTasks.TEST_LOOP.value:
            return self._get_loop_test_hierarchy()
        elif task_name == CraftWorldTasks.TEST_CONTEXT.value:
            return self._get_contextual_test_hierarchy()
        elif task_name == CraftWorldTasks.TEST_DISJUNCTION.value:
            return self._get_disjunction_test_hierarchy()
        elif task_name == CraftWorldTasks.TEST_DISJUNCTION_SIMPLE.value:
            return self._get_disjunction_simple_test_hierarchy()
        elif task_name == CraftWorldTasks.TEST_DISJUNCTION_DOUBLE.value:
            return self._get_disjunction_double_test_hierarchy()
        elif task_name == CraftWorldTasks.TEST_SIMULTANEOUS_SAT.value:
            return self._get_simultaneous_sat_test_hierarchy()

    def _get_flat_one_subgoal_hierarchy(self, task_name):
        subgoal = OBJ_TO_OBSERVABLE[task_name[len("get-"):]]

        root = HierarchicalAutomaton(CraftWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        root.add_state("u0")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        formula = [subgoal]
        if self.include_deadends and subgoal != CraftWorldObservables.LAVA:
            formula.append(f"~{CraftWorldObservables.LAVA}")

            root.add_state("u_rej")
            root.set_reject_state("u_rej")
            root.add_formula_edge("u0", "u_rej", DNFFormula([[CraftWorldObservables.LAVA]]))

        root.add_formula_edge("u0", "u_acc", DNFFormula([formula]))

        return self._build_hierarchy(root, [], [])

    def _get_flat_two_subgoal_hierarchy(self, task_name, subgoal1, subgoal2):
        return self._get_flat_hierarchy(task_name, 2, {
            ("u0", "u1"): [subgoal1],
            ("u1", "u_acc"): [subgoal2]
        })

    def _get_flat_diamond_hierarchy(self, task_name, subgoal1, subgoal2, subgoal3):
        return self._get_flat_hierarchy(task_name, 4, {
            ("u0", "u1"): [subgoal1],
            ("u0", "u2"): [f"~{subgoal1}", subgoal2],
            ("u1", "u3"): [subgoal2],
            ("u2", "u3"): [subgoal1],
            ("u3", "u_acc"): [subgoal3]
        })

    def _get_milk_bucket_hierarchy(self):
        return self._get_milk_bucket_flat_hierarchy() if self.use_flat_hierarchy else self._get_milk_bucket_nonflat_hierarchy()

    def _get_milk_bucket_flat_hierarchy(self):
        return self._get_flat_hierarchy(CraftWorldTasks.MILK_BUCKET.value, 3, {
            ("u0", "u1"): [CraftWorldObservables.IRON],
            ("u1", "u2"): [CraftWorldObservables.TABLE],
            ("u2", "u_acc"): [CraftWorldObservables.COW]
        })

    def _get_milk_bucket_nonflat_hierarchy(self):
        bucket_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.BUCKET.value)

        root = HierarchicalAutomaton(CraftWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[CraftWorldTasks.MILK_BUCKET.value])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")
        root.add_call_edge("u0", "u1", bucket_hierarchy.get_root_automaton().get_name(), TRUE)

        formula_u1_uacc = [CraftWorldObservables.COW]

        if self.include_deadends:
            formula_u1_uacc.append(f"~{CraftWorldObservables.LAVA}")

            root.add_state("u_rej")
            root.set_reject_state("u_rej")
            root.add_formula_edge("u1", "u_rej", DNFFormula([[CraftWorldObservables.LAVA]]))

        root.add_formula_edge("u1", "u_acc", DNFFormula([formula_u1_uacc]))

        return self._build_hierarchy(root, [bucket_hierarchy], [])

    def _get_map_hierarchy(self):
        return self._get_map_flat_hierarchy() if self.use_flat_hierarchy else self._get_map_nonflat_hierarchy()

    def _get_map_flat_hierarchy(self):
        return self._get_flat_hierarchy(CraftWorldTasks.BOOK.value, 11, {
            ("u0", "u1"): [CraftWorldObservables.SUGARCANE],
            ("u0", "u5"): [CraftWorldObservables.IRON, f"~{CraftWorldObservables.SUGARCANE}"],
            ("u0", "u6"): [CraftWorldObservables.REDSTONE, f"~{CraftWorldObservables.IRON}", f"~{CraftWorldObservables.SUGARCANE}"],
            ("u1", "u2"): [CraftWorldObservables.WORKBENCH],
            ("u2", "u3"): [CraftWorldObservables.IRON],
            ("u2", "u4"): [CraftWorldObservables.REDSTONE, f"~{CraftWorldObservables.IRON}"],
            ("u3", "u9"): [CraftWorldObservables.REDSTONE],
            ("u4", "u9"): [CraftWorldObservables.IRON],
            ("u5", "u7"): [CraftWorldObservables.REDSTONE],
            ("u6", "u7"): [CraftWorldObservables.IRON],
            ("u7", "u8"): [CraftWorldObservables.WORKBENCH],
            ("u8", "u9"): [CraftWorldObservables.SUGARCANE],
            ("u9", "u10"): [CraftWorldObservables.WORKBENCH],
            ("u10", "u_acc"): [CraftWorldObservables.TABLE]
        })

    def _get_map_nonflat_hierarchy(self):
        paper_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.PAPER.value)
        compass_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.COMPASS.value)

        root = HierarchicalAutomaton(CraftWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[CraftWorldTasks.MAP.value])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u2")
        root.add_state("u3")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        formula_u0_u2 = [f"~{CraftWorldObservables.SUGARCANE}"]
        formula_u3_uacc = [CraftWorldObservables.TABLE]

        if self.include_deadends:
            formula_u0_u2.append(f"~{CraftWorldObservables.LAVA}")  # a single path to rejection from the initial state
            formula_u3_uacc.append(f"~{CraftWorldObservables.LAVA}")

            root.add_state("u_rej")
            root.set_reject_state("u_rej")
            root.add_formula_edge("u3", "u_rej", DNFFormula([[CraftWorldObservables.LAVA]]))

        root.add_call_edge("u0", "u1", paper_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_call_edge("u0", "u2", compass_hierarchy.get_root_automaton().get_name(), DNFFormula([formula_u0_u2]))
        root.add_call_edge("u1", "u3", compass_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_call_edge("u2", "u3", paper_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_formula_edge("u3", "u_acc", DNFFormula([formula_u3_uacc]))

        return self._build_hierarchy(root, [paper_hierarchy, compass_hierarchy], [])

    def _get_book_hierarchy(self):
        return self._get_book_flat_hierarchy() if self.use_flat_hierarchy else self._get_book_nonflat_hierarchy()

    def _get_book_flat_hierarchy(self):
        return self._get_flat_hierarchy(CraftWorldTasks.BOOK.value, 7, {
            ("u0", "u1"): [CraftWorldObservables.SUGARCANE],
            ("u0", "u3"): [CraftWorldObservables.RABBIT, f"~{CraftWorldObservables.SUGARCANE}"],
            ("u1", "u2"): [CraftWorldObservables.WORKBENCH],
            ("u2", "u5"): [CraftWorldObservables.RABBIT],
            ("u3", "u4"): [CraftWorldObservables.WORKBENCH],
            ("u4", "u5"): [CraftWorldObservables.SUGARCANE],
            ("u5", "u6"): [CraftWorldObservables.WORKBENCH],
            ("u6", "u_acc"): [CraftWorldObservables.TABLE]
        })

    def _get_book_nonflat_hierarchy(self):
        paper_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.PAPER.value)
        leather_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.LEATHER.value)

        root = HierarchicalAutomaton(CraftWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[CraftWorldTasks.BOOK.value])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u2")
        root.add_state("u3")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        formula_u0_u2 = [f"~{CraftWorldObservables.SUGARCANE}"]
        formula_u3_uacc = [CraftWorldObservables.TABLE]

        if self.include_deadends:
            formula_u0_u2.append(f"~{CraftWorldObservables.LAVA}")  # a single path to rejection from the initial state
            formula_u3_uacc.append(f"~{CraftWorldObservables.LAVA}")

            root.add_state("u_rej")
            root.set_reject_state("u_rej")
            root.add_formula_edge("u3", "u_rej", DNFFormula([[CraftWorldObservables.LAVA]]))

        root.add_call_edge("u0", "u1", paper_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_call_edge("u0", "u2", leather_hierarchy.get_root_automaton().get_name(), DNFFormula([formula_u0_u2]))
        root.add_call_edge("u1", "u3", leather_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_call_edge("u2", "u3", paper_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_formula_edge("u3", "u_acc", DNFFormula([formula_u3_uacc]))

        return self._build_hierarchy(root, [paper_hierarchy, leather_hierarchy], [])

    def _get_book_and_quill_hierarchy(self):
        return self._get_book_and_quill_flat_hierarchy() if self.use_flat_hierarchy else self._get_book_and_quill_nonflat_hierarchy()

    def _get_book_and_quill_flat_hierarchy(self):
        return self._get_flat_hierarchy(CraftWorldTasks.BOOK_AND_QUILL.value, 20, {
            ("u0", "u1"): [CraftWorldObservables.SUGARCANE],
            ("u0", "u3"): [CraftWorldObservables.RABBIT, f"~{CraftWorldObservables.SUGARCANE}"],
            ("u0", "u11"): [CraftWorldObservables.CHICKEN, f"~{CraftWorldObservables.SUGARCANE}", f"~{CraftWorldObservables.RABBIT}"],
            ("u0", "u12"): [CraftWorldObservables.SQUID, f"~{CraftWorldObservables.CHICKEN}", f"~{CraftWorldObservables.SUGARCANE}", f"~{CraftWorldObservables.RABBIT}"],
            ("u1", "u2"): [CraftWorldObservables.WORKBENCH],
            ("u2", "u5"): [CraftWorldObservables.RABBIT],
            ("u3", "u4"): [CraftWorldObservables.WORKBENCH],
            ("u4", "u5"): [CraftWorldObservables.SUGARCANE],
            ("u5", "u6"): [CraftWorldObservables.WORKBENCH],
            ("u6", "u7"): [CraftWorldObservables.TABLE],
            ("u7", "u8"): [CraftWorldObservables.CHICKEN],
            ("u7", "u9"): [CraftWorldObservables.SQUID, f"~{CraftWorldObservables.CHICKEN}"],
            ("u8", "u10"): [CraftWorldObservables.SQUID],
            ("u9", "u10"): [CraftWorldObservables.CHICKEN],
            ("u11", "u13"): [CraftWorldObservables.SQUID],
            ("u12", "u13"): [CraftWorldObservables.CHICKEN],
            ("u13", "u14"): [CraftWorldObservables.TABLE],
            ("u14", "u15"): [CraftWorldObservables.SUGARCANE],
            ("u15", "u16"): [CraftWorldObservables.WORKBENCH],
            ("u16", "u19"): [CraftWorldObservables.RABBIT],
            ("u14", "u17"): [CraftWorldObservables.RABBIT, f"~{CraftWorldObservables.SUGARCANE}"],
            ("u17", "u18"): [CraftWorldObservables.WORKBENCH],
            ("u18", "u19"): [CraftWorldObservables.SUGARCANE],
            ("u19", "u10"): [CraftWorldObservables.WORKBENCH],
            ("u10", "u_acc"): [CraftWorldObservables.TABLE]
        })

    def _get_book_and_quill_nonflat_hierarchy(self):
        book_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.BOOK.value)
        quill_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.QUILL.value)

        root = HierarchicalAutomaton(CraftWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[CraftWorldTasks.BOOK_AND_QUILL.value])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u2")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        formula_u0_u2 = [f"~{CraftWorldObservables.SUGARCANE}", f"~{CraftWorldObservables.RABBIT}"]
        if self.include_deadends:
            formula_u0_u2.append(f"~{CraftWorldObservables.LAVA}")

        root.add_call_edge("u0", "u1", book_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_call_edge("u0", "u2", quill_hierarchy.get_root_automaton().get_name(), DNFFormula([formula_u0_u2]))
        root.add_call_edge("u1", "u_acc", quill_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_call_edge("u2", "u_acc", book_hierarchy.get_root_automaton().get_name(), TRUE)

        return self._build_hierarchy(root, [book_hierarchy, quill_hierarchy], [])

    def _get_milk_bucket_and_sugar_hierarchy(self):
        return self._get_milk_bucket_and_sugar_flat_hierarchy() if self.use_flat_hierarchy else self._get_milk_bucket_and_sugar_nonflat_hierarchy()

    def _get_milk_bucket_and_sugar_flat_hierarchy(self):
        return self._get_flat_hierarchy(CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value, 9, {
            ("u0", "u1"): [CraftWorldObservables.IRON, f"~{CraftWorldObservables.SUGARCANE}"],
            ("u0", "u5"): [CraftWorldObservables.SUGARCANE],
            ("u1", "u2"): [CraftWorldObservables.TABLE],
            ("u2", "u3"): [CraftWorldObservables.COW],
            ("u3", "u4"): [CraftWorldObservables.SUGARCANE],
            ("u4", "u_acc"): [CraftWorldObservables.TABLE],
            ("u5", "u6"): [CraftWorldObservables.TABLE],
            ("u6", "u7"): [CraftWorldObservables.IRON],
            ("u7", "u8"): [CraftWorldObservables.TABLE],
            ("u8", "u_acc"): [CraftWorldObservables.COW]
        })

    def _get_milk_bucket_and_sugar_nonflat_hierarchy(self):
        milk_bucket_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.MILK_BUCKET.value)
        sugar_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.SUGAR.value)

        root = HierarchicalAutomaton(CraftWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u2")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        formula_u0_u1 = [f"~{CraftWorldObservables.SUGARCANE}"]
        if self.include_deadends:
            formula_u0_u1.append(f"~{CraftWorldObservables.LAVA}")

        root.add_call_edge("u0", "u1", milk_bucket_hierarchy.get_root_automaton().get_name(), DNFFormula([formula_u0_u1]))
        root.add_call_edge("u0", "u2", sugar_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_call_edge("u1", "u_acc", sugar_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_call_edge("u2", "u_acc", milk_bucket_hierarchy.get_root_automaton().get_name(), TRUE)

        return self._build_hierarchy(root, [milk_bucket_hierarchy, sugar_hierarchy], [])

    def _get_cake_hierarchy(self):
        return self._get_cake_flat_hierarchy() if self.use_flat_hierarchy else self._get_cake_nonflat_hierarchy()

    def _get_cake_flat_hierarchy(self):
        return self._get_flat_hierarchy(CraftWorldTasks.CAKE.value, 14, {
            ("u0", "u1"): [CraftWorldObservables.WHEAT],
            ("u0", "u2"): [CraftWorldObservables.CHICKEN, f"~{CraftWorldObservables.WHEAT}"],
            ("u1", "u3"): [CraftWorldObservables.CHICKEN],
            ("u2", "u3"): [CraftWorldObservables.WHEAT],
            ("u3", "u4"): [CraftWorldObservables.TABLE],
            ("u4", "u5"): [CraftWorldObservables.IRON, f"~{CraftWorldObservables.SUGARCANE}"],
            ("u5", "u6"): [CraftWorldObservables.TABLE],
            ("u6", "u7"): [CraftWorldObservables.COW],
            ("u7", "u8"): [CraftWorldObservables.SUGARCANE],
            ("u8", "u13"): [CraftWorldObservables.TABLE],
            ("u4", "u9"): [CraftWorldObservables.SUGARCANE],
            ("u9", "u10"): [CraftWorldObservables.TABLE],
            ("u10", "u11"): [CraftWorldObservables.IRON],
            ("u11", "u12"): [CraftWorldObservables.TABLE],
            ("u12", "u13"): [CraftWorldObservables.COW],
            ("u13", "u_acc"): [CraftWorldObservables.WORKBENCH]
        })

    def _get_cake_nonflat_hierarchy(self):
        batter_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.BATTER.value)
        milk_bucket_and_sugar_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.MILK_BUCKET_AND_SUGAR.value)

        root = HierarchicalAutomaton(CraftWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[CraftWorldTasks.CAKE.value])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u2")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        formula_u2_uacc = [CraftWorldObservables.WORKBENCH]

        if self.include_deadends:
            formula_u2_uacc.append(f"~{CraftWorldObservables.LAVA}")

            root.add_state("u_rej")
            root.set_reject_state("u_rej")
            root.add_formula_edge("u2", "u_rej", DNFFormula([[CraftWorldObservables.LAVA]]))

        root.add_call_edge("u0", "u1", batter_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_call_edge("u1", "u2", milk_bucket_and_sugar_hierarchy.get_root_automaton().get_name(), TRUE)
        root.add_formula_edge("u2", "u_acc", DNFFormula([formula_u2_uacc]))

        return self._build_hierarchy(root, [batter_hierarchy, milk_bucket_and_sugar_hierarchy], [])

    def _get_flat_hierarchy(self, task_name, num_non_terminal_states, formula_edges):
        root = HierarchicalAutomaton(CraftWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        for i in range(num_non_terminal_states):
            root.add_state(f"u{i}")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        if self.include_deadends:
            for f in formula_edges:
                formula_edges[f].append(f"~{CraftWorldObservables.LAVA}")

            root.add_state("u_rej")
            root.set_reject_state("u_rej")
            for i in range(num_non_terminal_states):
                root.add_formula_edge(f"u{i}", "u_rej", DNFFormula([[CraftWorldObservables.LAVA]]))

        for f in formula_edges:
            root.add_formula_edge(f[0], f[1], DNFFormula([formula_edges[f]]))

        return self._build_hierarchy(root, [], [])

    def _get_ten_papers_hierarchy(self):
        return self._get_ten_papers_flat_hierarchy() if self.use_flat_hierarchy else self._get_ten_papers_nonflat_hierarchy()

    def _get_ten_papers_flat_hierarchy(self):
        states = [f"u{i}" for i in range(20)] + ["u_acc"]
        return self._get_flat_hierarchy(CraftWorldTasks.TEN_PAPERS.value, len(states) - 1, {
            (states[i], states[i + 1]): [CraftWorldObservables.SUGARCANE if i % 2 == 0 else CraftWorldObservables.WORKBENCH]
            for i in range(len(states) - 1)
        })

    def _get_ten_papers_nonflat_hierarchy(self):
        paper_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.PAPER.value)
        root = HierarchicalAutomaton(CraftWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[CraftWorldTasks.TEN_PAPERS.value])

        states = [f"u{i}" for i in range(10)] + ["u_acc"]
        for state in states:
            root.add_state(state)
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        for i in range(len(states) - 1):
            root.add_call_edge(states[i], states[i + 1], paper_hierarchy.get_root_automaton_name(), TRUE)

        return self._build_hierarchy(root, [paper_hierarchy], [])

    def _get_loop_test_hierarchy(self):
        """
        A dummy task with a loop to test the HRL algorithm (the contextual condition should become empty for u0 once we
        get back to it from u1, i.e. contextual conditions are important only after calls).
        """
        m0 = HierarchicalAutomaton("m0")
        m0.add_state("u0")
        m0.add_state("u1")
        m0.add_state("u2")
        m0.add_state("u_acc")
        m0.set_initial_state("u0")
        m0.set_accept_state("u_acc")
        m0.add_formula_edge("u0", "u1", DNFFormula([[CraftWorldObservables.IRON]]))
        m0.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.TABLE]]))
        m0.add_call_edge("u0", "u2", "m1", DNFFormula([[f"~{CraftWorldObservables.IRON}"]]))
        m0.add_formula_edge("u2", "u_acc", DNFFormula([[CraftWorldObservables.TABLE]]))

        m1 = HierarchicalAutomaton("m1")
        m1.add_state("u0")
        m1.add_state("u1")
        m1.add_state("u_acc")
        m1.set_initial_state("u0")
        m1.set_accept_state("u_acc")
        m1.add_formula_edge("u0", "u1", DNFFormula([[CraftWorldObservables.WHEAT]]))
        m1.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.SUGARCANE]]))
        m1.add_formula_edge("u1", "u0", DNFFormula([[CraftWorldObservables.CHICKEN,
                                                     f"~{CraftWorldObservables.SUGARCANE}"]]))

        return self._build_hierarchy(m0, [], [m1])

    def _get_contextual_test_hierarchy(self):
        """
        A task for testing the contexts: when the option is filled from the stack, the contextual conditions have to be
        added there too.
        """
        make_sugar_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.SUGAR.value)

        m0 = HierarchicalAutomaton("m0")
        m0.add_state("u0")
        m0.add_state("u1")
        m0.add_state("u2")
        m0.add_state("u_acc")
        m0.set_initial_state("u0")
        m0.set_accept_state("u_acc")
        m0.add_call_edge("u0", "u1", "m1", DNFFormula([[CraftWorldObservables.IRON]]))
        m0.add_call_edge("u0", "u2", "m1", DNFFormula([[f"~{CraftWorldObservables.IRON}"]]))
        m0.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.COW]]))
        m0.add_formula_edge("u2", "u_acc", DNFFormula([[CraftWorldObservables.SQUID]]))

        m1 = HierarchicalAutomaton("m1")
        m1.add_state("u0")
        m1.add_state("u1")
        m1.add_state("u2")
        m1.add_state("u_acc")
        m1.set_initial_state("u0")
        m1.set_accept_state("u_acc")
        m1.add_call_edge("u0", "u1", make_sugar_hierarchy.get_root_automaton().get_name(),
                         DNFFormula([[CraftWorldObservables.WHEAT]]))
        m1.add_call_edge("u0", "u2", make_sugar_hierarchy.get_root_automaton().get_name(),
                         DNFFormula([[f"~{CraftWorldObservables.WHEAT}"]]))
        m1.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.COW]]))
        m1.add_formula_edge("u2", "u_acc", DNFFormula([[CraftWorldObservables.SQUID]]))

        return self._build_hierarchy(m0, [make_sugar_hierarchy], [m1])

    def _get_disjunction_test_hierarchy(self):
        m0 = HierarchicalAutomaton("m0")
        m0.add_state("u0")
        m0.add_state("u_acc")
        m0.set_initial_state("u0")
        m0.set_accept_state("u_acc")
        m0.add_call_edge("u0", "u_acc", "m1", DNFFormula([[CraftWorldObservables.SUGARCANE], [CraftWorldObservables.IRON]]))
        m0.add_formula_edge("u0", "u_acc", DNFFormula([[CraftWorldObservables.SQUID, f"~{CraftWorldObservables.TABLE}"]]))

        m1 = HierarchicalAutomaton("m1")
        m1.add_state("u0")
        m1.add_state("u1")
        m1.add_state("u_acc")
        m1.set_initial_state("u0")
        m1.set_accept_state("u_acc")
        m1.add_call_edge("u0", "u1", "m2", DNFFormula([[CraftWorldObservables.COW]]))
        m1.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.RABBIT]]))
        m1.add_formula_edge("u1", "u0", DNFFormula([[CraftWorldObservables.SQUID, f"~{CraftWorldObservables.RABBIT}"]]))

        m2 = HierarchicalAutomaton("m2")
        m2.add_state("u0")
        m2.add_state("u_acc")
        m2.set_initial_state("u0")
        m2.set_accept_state("u_acc")
        m2.add_call_edge("u0", "u_acc", "m3", DNFFormula([[CraftWorldObservables.WHEAT]]))

        m3 = HierarchicalAutomaton("m3")
        m3.add_state("u0")
        m3.add_state("u1")
        m3.add_state("u_acc")
        m3.set_initial_state("u0")
        m3.set_accept_state("u_acc")
        m3.add_formula_edge("u0", "u1", DNFFormula([[CraftWorldObservables.TABLE]]))
        m3.add_formula_edge("u1", "u0", DNFFormula([[CraftWorldObservables.CHICKEN, f"~{CraftWorldObservables.REDSTONE}"]]))
        m3.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.REDSTONE]]))

        return self._build_hierarchy(m0, [], [m1, m2, m3])

    def _get_disjunction_simple_test_hierarchy(self):
        root = HierarchicalAutomaton("m0")
        root.add_state("u0")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")
        root.add_formula_edge("u0", "u_acc", DNFFormula([[CraftWorldObservables.SUGARCANE], [CraftWorldObservables.IRON]]))
        return self._build_hierarchy(root, [], [])

    def _get_disjunction_double_test_hierarchy(self):
        m0 = HierarchicalAutomaton("m0")
        m0.add_state("u0")
        m0.add_state("u1")
        m0.add_state("u_acc")
        m0.set_initial_state("u0")
        m0.set_accept_state("u_acc")
        m0.add_call_edge("u0", "u1", "m1", DNFFormula([[CraftWorldObservables.IRON], [CraftWorldObservables.TABLE]]))
        m0.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.COW]]))

        m1 = HierarchicalAutomaton("m1")
        m1.add_state("u0")
        m1.add_state("u1")
        m1.add_state("u_acc")
        m1.set_initial_state("u0")
        m1.set_accept_state("u_acc")
        m1.add_call_edge("u0", "u1", "m2",
                         DNFFormula([[CraftWorldObservables.SUGARCANE], [CraftWorldObservables.WHEAT]]))
        m1.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.COW]]))

        m2 = HierarchicalAutomaton("m2")
        m2.add_state("u0")
        m2.add_state("u1")
        m2.add_state("u_acc")
        m2.set_initial_state("u0")
        m2.set_accept_state("u_acc")
        m2.add_call_edge("u0", "u1", "m3", DNFFormula([[CraftWorldObservables.CHICKEN]]))
        m2.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.COW]]))

        m3 = HierarchicalAutomaton("m3")
        m3.add_state("u0")
        m3.add_state("u1")
        m3.add_state("u_acc")
        m3.set_initial_state("u0")
        m3.set_accept_state("u_acc")
        m3.add_formula_edge("u0", "u1", DNFFormula([[CraftWorldObservables.REDSTONE]]))
        m3.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.COW]]))

        return self._build_hierarchy(m0, [], [m1, m2, m3])

    def _get_simultaneous_sat_test_hierarchy(self):
        """
        A task for testing what happens when the local transition function of an automaton is satisfied by two calls
        that have been satisfied at once: here the edges from (u0, m0) are two calls that are going to be satisfied at
        the same time because m1 calls m2 and once m2 is satisfied, m1 will be as well!
        """
        sugar_hierarchy = self._get_hierarchy_for_task(CraftWorldTasks.SUGAR.value)

        m0 = HierarchicalAutomaton("m0")
        m0.add_state("u0")
        m0.add_state("u1")
        m0.add_state("u2")
        m0.add_state("u3")
        m0.add_state("u_acc")
        m0.set_initial_state("u0")
        m0.set_accept_state("u_acc")
        m0.add_call_edge("u0", "u1", "m1", DNFFormula([[f"~{CraftWorldObservables.SUGARCANE}"]]))
        m0.add_call_edge("u0", "u2", sugar_hierarchy.get_root_automaton().get_name(), TRUE)
        m0.add_formula_edge("u0", "u3", DNFFormula([[CraftWorldObservables.TABLE, f"~{CraftWorldObservables.SUGARCANE}",
                                                     f"~{CraftWorldObservables.COW}"]]))
        m0.add_formula_edge("u1", "u_acc", DNFFormula([[CraftWorldObservables.CHICKEN]]))
        m0.add_formula_edge("u2", "u_acc", DNFFormula([[CraftWorldObservables.SQUID]]))
        m0.add_formula_edge("u3", "u_acc", DNFFormula([[CraftWorldObservables.WHEAT]]))

        m1 = HierarchicalAutomaton("m1")
        m1.add_state("u0")
        m1.add_state("u1")
        m1.add_state("u_acc")
        m1.set_initial_state("u0")
        m1.set_accept_state("u_acc")
        m1.add_formula_edge("u0", "u1", DNFFormula([[CraftWorldObservables.COW]]))
        m1.add_call_edge("u1", "u_acc", sugar_hierarchy.get_root_automaton().get_name(), TRUE)

        return self._build_hierarchy(m0, [sugar_hierarchy], [m1])

    def env_reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode, highlight=False)

    def play(self):
        self.reset()
        self.game_window = Window("")
        self.game_window.reg_key_handler(self._key_handler)
        self._draw_game_window()
        self.game_window.show(block=True)

    def _key_handler(self, event):
        if event.key == "escape":
            self.game_window.close()
            return

        if event.key == "backspace":
            return

        action = -1
        if event.key == "left":
            action = 0
        elif event.key == "right":
            action = 1
        elif event.key == "up":
            action = 2

        if action >= 0:
            obs, reward, is_terminal, info = self.step(action)
            # print(f"State: {obs}, Position: {self.env.agent_pos}, Direction: {self.env.agent_dir}, Reward: {reward}, "
            #       f"Observation: {self.get_observation()}")

            if is_terminal:
                print(f"Reward: {reward}")
                self.game_window.close()
                return
            else:
                self._draw_game_window()

    def _draw_game_window(self):
        self.game_window.show_img(self.render("rgb_array"))
