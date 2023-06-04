from gym import spaces
from gym_hierarchical_subgoal_automata.automata.common import get_param
from gym_hierarchical_subgoal_automata.automata.hierarchical_automaton import HierarchicalAutomaton
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula, TRUE
from gym_hierarchical_subgoal_automata.envs.base.base_env import BaseEnv, TaskEnum
from itertools import chain, combinations
import math
import numpy as np
import pygame
import random
import time
from typing import List


def powerset(iterable):
    # Copied from https://docs.python.org/2/library/itertools.html#recipes
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class WaterWorldActions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NONE = 4


class WaterWorldObservables:
    RED = "r"
    GREEN = "g"
    CYAN = "c"
    BLUE = "b"
    YELLOW = "y"
    MAGENTA = "m"
    EMPTY = "e"
    BLACK = "k"


class WaterWorldTasks(TaskEnum):
    # A single colour
    R = WaterWorldObservables.RED
    G = WaterWorldObservables.GREEN
    B = WaterWorldObservables.BLUE
    C = WaterWorldObservables.CYAN
    Y = WaterWorldObservables.YELLOW
    M = WaterWorldObservables.MAGENTA

    # Two element sequences (X-Y = X then Y)
    RG = f"{R}-{G}"
    BC = f"{B}-{C}"
    MY = f"{M}-{Y}"
    RE = f"{R}-{WaterWorldObservables.EMPTY}"
    GE = f"{G}-{WaterWorldObservables.EMPTY}"
    BE = f"{B}-{WaterWorldObservables.EMPTY}"
    CE = f"{C}-{WaterWorldObservables.EMPTY}"
    YE = f"{Y}-{WaterWorldObservables.EMPTY}"
    ME = f"{M}-{WaterWorldObservables.EMPTY}"

    # Three element sequences (X-Y-Z = X then Y then Z)
    RGB = f"{R}-{G}-{B}"
    CMY = f"{C}-{M}-{Y}"

    # Interleaving sequences (X&Y = X then Y OR Y then X) -> elements can be done in any order
    RG_BC = f"({RG})&({BC})"
    BC_MY = f"({BC})&({MY})"
    RG_MY = f"({RG})&({MY})"
    RG_BC_MY = f"({RG})&({BC})&({MY})"
    RGB_CMY = f"({RGB})&({CMY})"

    # Build up of the avoidance of all colours (COLOUR_WO_COLOURS)
    R_WO_G, R_WO_GB, R_WO_GBC, R_WO_GBCY, R_WO_GBCYM = f"{R}\\{G}", f"{R}\\{G}{B}", f"{R}\\{G}{B}{C}", f"{R}\\{G}{B}{C}{Y}", f"{R}\\{G}{B}{C}{Y}{M}"
    G_WO_B, G_WO_BC, G_WO_BCY, G_WO_BCYM, G_WO_BCYMR = f"{G}\\{B}", f"{G}\\{B}{C}", f"{G}\\{B}{C}{Y}", f"{G}\\{B}{C}{Y}{M}", f"{G}\\{B}{C}{Y}{M}{R}"
    B_WO_C, B_WO_CY, B_WO_CYM, B_WO_CYMR, B_WO_CYMRG = f"{B}\\{C}", f"{B}\\{C}{Y}", f"{B}\\{C}{Y}{M}", f"{B}\\{C}{Y}{M}{R}", f"{B}\\{C}{Y}{M}{R}{G}"
    C_WO_Y, C_WO_YM, C_WO_YMR, C_WO_YMRG, C_WO_YMRGB = f"{C}\\{Y}", f"{C}\\{Y}{M}", f"{C}\\{Y}{M}{R}", f"{C}\\{Y}{M}{R}{G}", f"{C}\\{Y}{M}{R}{G}{B}"
    Y_WO_M, Y_WO_MR, Y_WO_MRG, Y_WO_MRGB, Y_WO_MRGBC = f"{Y}\\{M}", f"{Y}\\{M}{R}", f"{Y}\\{M}{R}{G}", f"{Y}\\{M}{R}{G}{B}", f"{Y}\\{M}{R}{G}{B}{C}"
    M_WO_R, M_WO_RG, M_WO_RGB, M_WO_RGBC, M_WO_RGBCY = f"{M}\\{R}", f"{M}\\{R}{G}", f"{M}\\{R}{G}{B}", f"{M}\\{R}{G}{B}{C}", f"{M}\\{R}{G}{B}{C}{Y}"
    RGB_FULL_STRICT = f"({R_WO_GBCYM})-({G_WO_BCYMR})-({B_WO_CYMRG})"
    CMY_FULL_STRICT = f"({C_WO_YMRGB})-({M_WO_RGBCY})-({Y_WO_MRGBC})"

    # Build up of the avoidance of colors in the same group (RGB and CYM). Ex: Avoid G and B while pursuing R.
    # R_WO_G, R_WO_GB, G_WO_B, C_WO_Y, C_WO_YM and Y_WO_M are defined above
    G_WO_BR = f"{G}\\{B}{R}"
    B_WO_R, B_WO_RG = f"{B}\\{R}", f"{B}\\{R}{G}"
    Y_WO_MC = f"{Y}\\{M}{C}"
    M_WO_C, M_WO_CY = f"{M}\\{C}", f"{M}\\{C}{Y}"
    RGB_INTERAVOIDANCE = f"({R_WO_GB})-({G_WO_BR})-({B_WO_RG})"
    CMY_INTERAVOIDANCE = f"({C_WO_YM})-({M_WO_CY})-({Y_WO_MC})"
    RGB_CMY_INTERAVOIDANCE = f"({RGB_INTERAVOIDANCE})&({CMY_INTERAVOIDANCE})"

    # Avoidance in first step of a colour followed by 'empty' sequence. Ex: RE_WO_G consists in observing red while
    # avoiding green, then observe empty (no color).
    RE_WO_G, RE_WO_GB = f"({R}\\{G})-{WaterWorldObservables.EMPTY}", f"({R}\\{G}{B})-{WaterWorldObservables.EMPTY}"
    GE_WO_B, GE_WO_BR = f"({G}\\{B})-{WaterWorldObservables.EMPTY}", f"({G}\\{B}{R})-{WaterWorldObservables.EMPTY}"
    BE_WO_R, BE_WO_RG = f"({B}\\{R})-{WaterWorldObservables.EMPTY}", f"({B}\\{R}{G})-{WaterWorldObservables.EMPTY}"
    CE_WO_M, CE_WO_MY = f"({C}\\{M})-{WaterWorldObservables.EMPTY}", f"({C}\\{M}{Y})-{WaterWorldObservables.EMPTY}"
    ME_WO_Y, ME_WO_YC = f"({M}\\{Y})-{WaterWorldObservables.EMPTY}", f"({M}\\{Y}{C})-{WaterWorldObservables.EMPTY}"
    YE_WO_C, YE_WO_CM = f"({Y}\\{C})-{WaterWorldObservables.EMPTY}", f"({Y}\\{C}{M})-{WaterWorldObservables.EMPTY}"
    REGEBE_INTERAVOIDANCE = f"({RE_WO_GB})-({GE_WO_BR})-({BE_WO_RG})"
    CEMEYE_INTERAVOIDANCE = f"({CE_WO_MY})-({ME_WO_YC})-({YE_WO_CM})"
    REGEBE_CEMEYE_INTERAVOIDANCE = f"({REGEBE_INTERAVOIDANCE})&({CEMEYE_INTERAVOIDANCE})"

    # Avoid the next two colours (in the chain R,G,B,C,M,Y - a given colour has to avoid the next two in the chain)
    GE_WO_BC = f"({G}\\{B}{C})-{WaterWorldObservables.EMPTY}"
    BE_WO_C, BE_WO_CM = f"({B}\\{C})-{WaterWorldObservables.EMPTY}", f"({B}\\{C}{M})-{WaterWorldObservables.EMPTY}"
    ME_WO_YR = f"({M}\\{Y}{R})-{WaterWorldObservables.EMPTY}"
    YE_WO_R, YE_WO_RG = f"({Y}\\{R})-{WaterWorldObservables.EMPTY}", f"({Y}\\{R}{G})-{WaterWorldObservables.EMPTY}"
    REGEBE_AVOID_NEXT_TWO = f"({RE_WO_GB})-({GE_WO_BC})-({BE_WO_CM})"
    CEMEYE_AVOID_NEXT_TWO = f"({CE_WO_MY})-({ME_WO_YR})-({YE_WO_RG})"
    REGEBE_CEMEYE_AVOID_NEXT_TWO = f"({REGEBE_AVOID_NEXT_TWO})&({CEMEYE_AVOID_NEXT_TWO})"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class Ball:
    def __init__(self, color, radius, pos, vel):
        self.color = color
        self.radius = radius
        self.pos = None
        self.vel = None
        self.update(pos, vel)

    def __str__(self):
        return "\t".join([self.color, str(self.pos[0]), str(self.pos[1]), str(self.vel[0]), str(self.vel[1])])

    def update_position(self, elapsed_time):
        self.pos += elapsed_time * self.vel

    def update(self, pos, vel):
        self.pos = np.array(pos, dtype=np.float)
        self.vel = np.array(vel, dtype=np.float)

    def is_colliding(self, ball):
        d = np.linalg.norm(self.pos - ball.pos, ord=2)
        return d <= self.radius + ball.radius


class BallAgent(Ball):
    def __init__(self, color, radius, pos, vel, vel_delta, vel_max):
        super().__init__(color, radius, pos, vel)
        self.vel_delta = float(vel_delta)
        self.vel_max = float(vel_max)

    def step(self, action):
        # updating velocity
        delta = np.array([0, 0])
        if action == WaterWorldActions.UP:
            delta = np.array([0.0, +1.0])
        elif action == WaterWorldActions.DOWN:
            delta = np.array([0.0, -1.0])
        elif action == WaterWorldActions.LEFT:
            delta = np.array([-1.0, 0.0])
        elif action == WaterWorldActions.RIGHT:
            delta = np.array([+1.0, 0.0])

        self.vel += self.vel_delta * delta

        # checking limits
        self.vel = np.clip(self.vel, -self.vel_max, self.vel_max)


class WaterWorldEnv(BaseEnv):
    """
    The Water World environment
    from "Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning"
    by Rodrigo Toro Icarte, Toryn Q. Klassen, Richard Valenzano and Sheila A. McIlraith.

    Description:
    It consists of a 2D box containing 12 balls of different colors (2 balls per color). Each ball moves at a constant
    speed in a given direction and bounces when it collides with a wall. The agent is a white ball that can change its
    velocity in any of the four cardinal directions.

    Rewards:
    Different tasks (subclassed below) are defined in this environment. All of them are goal-oriented, i.e., provide
    a reward of 1 when a certain goal is achieved and 0 otherwise. The goal always consists of touching a sequence of
    balls in a specific order.

    Actions:
    - 0: up
    - 1: down
    - 2: left
    - 3: right
    - 4: none

    Acknowledgments:
    Most of the code has been reused from the original implementation by the authors of reward machines:
    https://bitbucket.org/RToroIcarte/qrm/src/master/.
    """
    RENDERING_COLORS = {
        "A": (0, 0, 0),
        WaterWorldObservables.RED: (255, 0, 0),
        WaterWorldObservables.GREEN: (0, 255, 0),
        WaterWorldObservables.BLUE: (0, 0, 255),
        WaterWorldObservables.YELLOW: (255, 255, 0),
        WaterWorldObservables.CYAN: (0, 255, 255),
        WaterWorldObservables.MAGENTA: (255, 0, 255),
        WaterWorldObservables.BLACK: (0, 0, 0)
    }

    TASK_NAME_TO_AUTOMATON_NAME = {
        task_name: f"m{index}"
        for index, task_name in enumerate(WaterWorldTasks.list())
    }

    MAX_X = "max_x"
    MAX_Y = "max_y"
    BALL_RADIUS = "ball_radius"
    BALL_VELOCITY = "ball_velocity"
    BALLS_PER_COLOR = "ball_num_per_color"
    USE_VELOCITIES = "use_velocities"
    USE_EMPTY = "use_empty"
    AVOID_BLACK = "avoid_black"
    NUM_BLACK_BALLS = "num_black_balls"

    def __init__(self, task_name, params):
        super().__init__(params)

        self.task_name = task_name

        # Random generator used to generate the instances
        self.random_gen = None

        # Parameters
        self.max_x = get_param(params, WaterWorldEnv.MAX_X, 400)
        self.max_y = get_param(params, WaterWorldEnv.MAX_Y, 400)
        self.ball_radius = get_param(params, WaterWorldEnv.BALL_RADIUS, 15)
        self.ball_velocity = get_param(params, WaterWorldEnv.BALL_VELOCITY, 30)
        self.ball_num_per_color = get_param(params, WaterWorldEnv.BALLS_PER_COLOR, 2)
        self.use_velocities = get_param(params, WaterWorldEnv.USE_VELOCITIES, True)
        self.use_empty = get_param(params, WaterWorldEnv.USE_EMPTY, False)
        self.avoid_black = get_param(params, WaterWorldEnv.AVOID_BLACK, False)
        if self.avoid_black:
            self.num_black_balls = get_param(params, WaterWorldEnv.NUM_BLACK_BALLS, self.ball_num_per_color)
        self.ball_num_colors = len(self.get_observables()) - 1 if self.use_empty else len(self.get_observables())
        self.agent_vel_delta = self.ball_velocity
        self.agent_vel_max = 3 * self.ball_velocity

        # Agent ball and other balls to avoid or touch
        self.agent = None
        self.balls = []

        self.observation_space = spaces.Box(low=-2, high=2, shape=(self._get_feature_size(),), dtype=np.float)
        self.action_space = spaces.Discrete(5)

        # Rendering attributes
        self.is_rendering = False
        self.game_display = None

    def _get_pos_vel_new_ball(self):
        angle = self.random_gen.random() * 2 * math.pi

        if self.use_velocities:
            vel = self.ball_velocity * math.sin(angle), self.ball_velocity * math.cos(angle)
        else:
            vel = 0.0, 0.0

        while True:
            pos = 2 * self.ball_radius + self.random_gen.random() * (self.max_x - 2 * self.ball_radius), \
                  2 * self.ball_radius + self.random_gen.random() * (self.max_y - 2 * self.ball_radius)
            if not self._is_colliding(pos) and np.linalg.norm(self.agent.pos - np.array(pos), ord=2) > 4 * self.ball_radius:
                break
        return pos, vel

    def _is_colliding(self, pos):
        for b in self.balls + [self.agent]:
            if np.linalg.norm(b.pos - np.array(pos), ord=2) < 2 * self.ball_radius:
                return True
        return False

    def env_step(self, action, elapsed_time=0.1):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        if self.is_terminal():
            return self._get_features(), 0.0, True, self.get_observation()

        # Updating the agents velocity
        self.agent.step(action)
        balls_all = [self.agent] + self.balls
        max_x, max_y = self.max_x, self.max_y

        # Updating position
        for b in balls_all:
            b.update_position(elapsed_time)

        # Handling collisions
        for i in range(len(balls_all)):
            b = balls_all[i]
            # Walls
            if b.pos[0] - b.radius < 0 or b.pos[0] + b.radius > max_x:
                # Place ball against edge
                if b.pos[0] - b.radius < 0:
                    b.pos[0] = b.radius
                else:
                    b.pos[0] = max_x - b.radius
                # Reverse direction
                b.vel *= np.array([-1.0, 1.0])
            if b.pos[1] - b.radius < 0 or b.pos[1] + b.radius > max_y:
                # Place ball against edge
                if b.pos[1] - b.radius < 0:
                    b.pos[1] = b.radius
                else:
                    b.pos[1] = max_y - b.radius
                # Reverse direction
                b.vel *= np.array([1.0, -1.0])

        return self._get_features(), {}

    def get_observables(self):
        observables = [WaterWorldObservables.RED, WaterWorldObservables.GREEN, WaterWorldObservables.BLUE,
                       WaterWorldObservables.CYAN, WaterWorldObservables.MAGENTA, WaterWorldObservables.YELLOW]
        if self.avoid_black:
            observables.append(WaterWorldObservables.BLACK)
        if self.use_empty:
            observables.append(WaterWorldObservables.EMPTY)
        return observables

    def get_restricted_observables(self) -> List[str]:
        return self._get_restricted_observables_for_task(self.task_name)

    def _get_restricted_observables_for_task(self, task_name):
        if self._is_n_subgoal_task(task_name):
            return task_name.split("-") + [WaterWorldObservables.BLACK] if self.avoid_black else []
        elif self._is_two_anyorder_task(task_name):
            st1, st2 = task_name.split("&")
            return self._get_restricted_observables_from_dependencies(
                [WaterWorldObservables.BLACK] if self.avoid_black else [],
                [st1[1:-1], st2[1:-1]]
            )
        elif task_name == WaterWorldTasks.RG_BC_MY.value:
            return self._get_restricted_observables_from_dependencies(
                [WaterWorldObservables.BLACK] if self.avoid_black else [],
                [WaterWorldTasks.RG.value, WaterWorldTasks.BC.value, WaterWorldTasks.MY.value]
            )
        elif self._is_simple_avoidance_task(task_name):
            task_name_split = task_name.split("\\")
            return [task_name_split[0]] + list(task_name_split[1])
        elif self._is_avoidance_with_empty_task(task_name):
            return self._get_restricted_observables_from_dependencies([WaterWorldObservables.EMPTY], [task_name.split("-")[0][1:-1]])
        elif self._is_n_call_sequence_hierarchy(task_name):
            task_name_split = task_name[1:-1].split(")-(")
            return self._get_restricted_observables_from_dependencies([], task_name_split)
        elif task_name == WaterWorldTasks.RGB_CMY_INTERAVOIDANCE.value:
            return self._get_restricted_observables_from_dependencies([], [WaterWorldTasks.RGB_INTERAVOIDANCE.value, WaterWorldTasks.CMY_INTERAVOIDANCE.value])
        elif task_name == WaterWorldTasks.REGEBE_CEMEYE_INTERAVOIDANCE.value:
            return self._get_restricted_observables_from_dependencies([], [WaterWorldTasks.REGEBE_INTERAVOIDANCE.value, WaterWorldTasks.CEMEYE_INTERAVOIDANCE.value])
        elif task_name == WaterWorldTasks.REGEBE_CEMEYE_AVOID_NEXT_TWO.value:
            return self._get_restricted_observables_from_dependencies([], [WaterWorldTasks.REGEBE_AVOID_NEXT_TWO.value, WaterWorldTasks.CEMEYE_AVOID_NEXT_TWO.value])

    def _get_restricted_observables_from_dependencies(self, base_observables, dependencies):
        observables = set(base_observables)
        for dependency in dependencies:
            observables.update(self._get_restricted_observables_for_task(dependency))
        return list(observables)

    def get_observation(self):
        collisions = self._get_current_collisions()
        if self.use_empty and len(collisions) == 0:
            return {WaterWorldObservables.EMPTY}
        return {b.color for b in self._get_current_collisions()}

    def get_possible_observations(self):
        observations = []
        colors = [
            WaterWorldObservables.RED, WaterWorldObservables.GREEN, WaterWorldObservables.BLUE,
            WaterWorldObservables.CYAN, WaterWorldObservables.MAGENTA, WaterWorldObservables.YELLOW
        ]
        if self.avoid_black:
            colors.append(WaterWorldObservables.BLACK)

        for observation in powerset(colors):
            # Exclude the empty observation if we use a proposition for it (added later).
            if len(observation) > 0 or not self.use_empty:
                observations.append(set(observation))
        if self.use_empty:
            observations.append(set(WaterWorldObservables.EMPTY))
        observations.sort(key=lambda x: tuple(x))
        return observations

    def _get_current_collisions(self):
        collisions = set()
        for b in self.balls:
            if self.agent.is_colliding(b):
                collisions.add(b)
        return collisions

    def get_hierarchy(self):
        return self._get_hierarchy_for_task(self.task_name)

    def _get_hierarchy_for_task(self, task_name):
        if self._is_n_subgoal_task(task_name):
            return self._get_n_subgoal_hierarchy(task_name)
        elif self._is_two_anyorder_task(task_name):
            return self._get_anyorder_hierarchy(task_name)
        elif task_name == WaterWorldTasks.RG_BC_MY.value:
            return self._get_rg_bc_my_hierarchy(task_name)
        elif self._is_simple_avoidance_task(task_name):
            return self._get_simple_avoidance_hierarchy(task_name)
        elif self._is_avoidance_with_empty_task(task_name):
            return self._get_avoidance_with_empty_hierarchy(task_name)
        elif self._is_n_call_sequence_hierarchy(task_name):
            return self._get_n_call_sequence_hierarchy(task_name)
        elif task_name == WaterWorldTasks.RGB_CMY_INTERAVOIDANCE.value:
            return self._get_rgb_cym_interavoidance_hierarchy(task_name, False)
        elif task_name == WaterWorldTasks.REGEBE_CEMEYE_INTERAVOIDANCE.value:
            return self._get_rgb_cym_interavoidance_hierarchy(task_name, True)
        elif task_name == WaterWorldTasks.REGEBE_CEMEYE_AVOID_NEXT_TWO.value:
            return self._get_rgb_cym_avoid_next_two_hierarchy(task_name)

    def _is_n_subgoal_task(self, task_name):
        """
        Returns True if the task consists in observing a sequence of observables in a specific order.
        """
        subgoals = task_name.split("-")
        if len(subgoals) > 0:
            return all(len(s) == 1 for s in subgoals)
        return False

    def _is_two_anyorder_task(self, task_name):
        subtasks = task_name.split("&")
        if len(subtasks) == 2:
            return all(self._is_n_subgoal_task(s[1:-1]) for s in subtasks)
        return False

    def _is_simple_avoidance_task(self, task_name):
        decomp = task_name.split("\\")
        return len(decomp) == 2 and len(decomp[0]) == 1

    def _is_avoidance_with_empty_task(self, task_name):
        decomp = task_name.split("-")
        return len(decomp) == 2 and self._is_simple_avoidance_task(decomp[0][1:-1]) and decomp[1] == WaterWorldObservables.EMPTY

    def _is_n_call_sequence_hierarchy(self, task_name):
        if "&" not in task_name:
            return len(task_name.split(")-(")) > 0
        return False

    def _get_n_subgoal_hierarchy(self, task_name):
        root = HierarchicalAutomaton(WaterWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        subgoals = task_name.split("-")

        for i in range(len(subgoals)):
            root.add_state(f"u{i}")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        if self.avoid_black:
            root.add_state("u_rej")
            root.set_reject_state("u_rej")

        def _add_edge_n_subgoal_hierarchy(state_id, subgoal):
            from_state = f"u{state_id}"
            if state_id == len(subgoals) - 1:
                to_state = "u_acc"
            else:
                to_state = f"u{state_id + 1}"

            formula = [subgoal]
            if self.avoid_black and subgoal != WaterWorldObservables.BLACK:
                formula.append(f"~{WaterWorldObservables.BLACK}")
                root.add_formula_edge(from_state, "u_rej", DNFFormula([[WaterWorldObservables.BLACK]]))
            root.add_formula_edge(from_state, to_state, DNFFormula([formula]))

        count = 0
        while count < len(subgoals) - 1:
            _add_edge_n_subgoal_hierarchy(count, subgoals[count])
            count += 1
        _add_edge_n_subgoal_hierarchy(count, subgoals[count])

        return self._build_hierarchy(root, [], [])

    def _get_anyorder_hierarchy(self, task_name):
        return self._get_anyorder_flat_hierarchy(task_name) if self.use_flat_hierarchy else self._get_anyorder_nonflat_hierarchy(task_name)

    def _get_anyorder_flat_hierarchy(self, task_name):
        st1, st2 = task_name.split("&")
        st1, st2 = st1[1:-1], st2[1:-1]
        st1_subgoals = st1.split("-")
        st2_subgoals = st2.split("-")

        root = HierarchicalAutomaton(WaterWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        root.add_state("u0")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        if self.avoid_black:
            root.add_state("u_rej")
            root.set_reject_state("u_rej")

        current_id = 1

        def add_edges(start_id, sequence, alt_sequence):
            for i in range(len(sequence)):
                formula = [sequence[i]]

                if i == 0:
                    from_state = "u0"
                    formula.append(f"~{alt_sequence[i]}")
                else:
                    from_state = f"u{start_id - 1}"

                if i + 1 == len(sequence):
                    to_state = "u_acc"
                else:
                    to_state = f"u{start_id}"
                    start_id += 1
                    root.add_state(to_state)

                if self.avoid_black:
                    formula.append(f"~{WaterWorldObservables.BLACK}")

                root.add_formula_edge(from_state, to_state, DNFFormula([formula]))
            return start_id

        current_id = add_edges(current_id, [*st1_subgoals, *st2_subgoals], st2_subgoals)
        current_id = add_edges(current_id, [*st2_subgoals, *st1_subgoals], st1_subgoals)

        if self.avoid_black:
            for state_id in range(0, current_id):
                root.add_formula_edge(f"u{state_id}", "u_rej", DNFFormula([[WaterWorldObservables.BLACK]]))

        return self._build_hierarchy(root, [], [])

    def _get_anyorder_nonflat_hierarchy(self, task_name):
        st1, st2 = task_name.split("&")
        st1, st2 = st1[1:-1], st2[1:-1]
        st1_hierarchy, st2_hierarchy = self._get_hierarchy_for_task(st1), self._get_hierarchy_for_task(st2)

        root = HierarchicalAutomaton(WaterWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u2")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        formula_u0_u1 = [f"~{st2[0]}"]
        formula_u0_u2 = [f"~{st1[0]}"]

        if self.avoid_black:
            root.add_state("u_rej")
            root.set_reject_state("u_rej")
            root.add_formula_edge("u0", "u_rej", DNFFormula([[WaterWorldObservables.BLACK]]))

            for formula in [formula_u0_u1, formula_u0_u2]:
                formula.append(f"~{WaterWorldObservables.BLACK}")

        root.add_call_edge("u0", "u1", st1_hierarchy.get_root_automaton_name(), DNFFormula([formula_u0_u1]))
        root.add_call_edge("u0", "u2", st2_hierarchy.get_root_automaton_name(), DNFFormula([formula_u0_u2]))
        root.add_call_edge("u1", "u_acc", st2_hierarchy.get_root_automaton_name(), TRUE)
        root.add_call_edge("u2", "u_acc", st1_hierarchy.get_root_automaton_name(), TRUE)

        return self._build_hierarchy(root, [st1_hierarchy, st2_hierarchy], [])

    def _get_rg_bc_my_hierarchy(self, task_name):
        rg_hierarchy = self._get_hierarchy_for_task(WaterWorldTasks.RG.value)
        bc_hierarchy = self._get_hierarchy_for_task(WaterWorldTasks.BC.value)
        my_hierarchy = self._get_hierarchy_for_task(WaterWorldTasks.MY.value)

        root = HierarchicalAutomaton(WaterWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u2")
        root.add_state("u3")
        root.add_state("u4")
        root.add_state("u5")
        root.add_state("u6")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        formula_u0_u1 = [f"~{WaterWorldObservables.BLUE}", f"~{WaterWorldObservables.MAGENTA}"]
        formula_u0_u2 = [f"~{WaterWorldObservables.RED}", f"~{WaterWorldObservables.MAGENTA}"]
        formula_u0_u3 = [f"~{WaterWorldObservables.RED}", f"~{WaterWorldObservables.BLUE}"]

        formula_u2_u4 = [f"~{WaterWorldObservables.MAGENTA}"]
        formula_u2_u5 = [f"~{WaterWorldObservables.RED}"]

        formula_u3_u5 = [f"~{WaterWorldObservables.RED}"]
        formula_u3_u6 = [f"~{WaterWorldObservables.BLUE}"]

        if self.avoid_black:
            root.add_state("u_rej")
            root.set_reject_state("u_rej")

            for from_state in ["u0", "u2", "u3"]:
                root.add_formula_edge(from_state, "u_rej", DNFFormula([[WaterWorldObservables.BLACK]]))

            for formula in [formula_u0_u1, formula_u0_u2, formula_u0_u3, formula_u2_u4, formula_u2_u5, formula_u3_u5, formula_u3_u6]:
                formula.append(f"~{WaterWorldObservables.BLACK}")

        root.add_call_edge("u0", "u1", rg_hierarchy.get_root_automaton_name(), DNFFormula([formula_u0_u1]))
        root.add_call_edge("u0", "u2", bc_hierarchy.get_root_automaton_name(), DNFFormula([formula_u0_u2]))
        root.add_call_edge("u0", "u3", my_hierarchy.get_root_automaton_name(), DNFFormula([formula_u0_u3]))

        root.add_call_edge("u1", "u4", bc_hierarchy.get_root_automaton_name(), TRUE)

        root.add_call_edge("u2", "u4", rg_hierarchy.get_root_automaton_name(), DNFFormula([formula_u2_u4]))
        root.add_call_edge("u2", "u5", my_hierarchy.get_root_automaton_name(), DNFFormula([formula_u2_u5]))

        root.add_call_edge("u3", "u5", bc_hierarchy.get_root_automaton_name(), DNFFormula([formula_u3_u5]))
        root.add_call_edge("u3", "u6", rg_hierarchy.get_root_automaton_name(), DNFFormula([formula_u3_u6]))

        root.add_call_edge("u4", "u_acc", my_hierarchy.get_root_automaton_name(), TRUE)
        root.add_call_edge("u5", "u_acc", rg_hierarchy.get_root_automaton_name(), TRUE)
        root.add_call_edge("u6", "u_acc", bc_hierarchy.get_root_automaton_name(), TRUE)

        return self._build_hierarchy(root, [rg_hierarchy, bc_hierarchy, my_hierarchy], [])

    def _get_simple_avoidance_hierarchy(self, task_name):
        base_objective, avoid = task_name.split("\\")
        if len(avoid) == 1:
            called_task_name = base_objective
        else:
            called_task_name = f"{base_objective}\\{avoid[:-1]}"
        return self._get_avoidance_hierarchy(task_name, self._get_hierarchy_for_task(called_task_name), avoid[-1])

    def _get_avoidance_with_empty_hierarchy(self, task_name):
        avoidance_task = task_name.split("-")[0][1:-1]
        base_objective, avoid = avoidance_task.split("\\")
        if len(avoid) == 1:
            called_task_name = f"{base_objective}-{WaterWorldObservables.EMPTY}"
        else:
            called_task_name = f"({base_objective}\\{avoid[:-1]})-{WaterWorldObservables.EMPTY}"
        return self._get_avoidance_hierarchy(task_name, self._get_hierarchy_for_task(called_task_name), avoid[-1])

    def _get_avoidance_hierarchy(self, task_name, called_hierarchy, to_avoid):
        root = HierarchicalAutomaton(WaterWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        root.add_state("u0")
        root.add_state("u_acc")
        root.add_state("u_rej")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")
        root.set_reject_state("u_rej")
        root.add_call_edge("u0", "u_acc", called_hierarchy.get_root_automaton_name(), DNFFormula([[f"~{to_avoid}"]]))
        root.add_formula_edge("u0", "u_rej", DNFFormula([[to_avoid]]))
        return self._build_hierarchy(root, [called_hierarchy], [])

    def _get_n_call_sequence_hierarchy(self, task_name):
        root = HierarchicalAutomaton(WaterWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        subtasks = task_name[1:-1].split(")-(")

        for i in range(len(subtasks)):
            root.add_state(f"u{i}")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")

        dependency_hierarchies = []
        count = 0
        while count < len(subtasks) - 1:
            dependency_hierarchies.append(self._get_hierarchy_for_task(subtasks[count]))
            root.add_call_edge(f"u{count}", f"u{count + 1}", dependency_hierarchies[-1].get_root_automaton_name(), TRUE)
            count += 1
        dependency_hierarchies.append(self._get_hierarchy_for_task(subtasks[count]))
        root.add_call_edge(f"u{count}", "u_acc", dependency_hierarchies[-1].get_root_automaton_name(), TRUE)

        return self._build_hierarchy(root, dependency_hierarchies, [])

    def _get_three_sequence_hierarchy(self, task_name, task_1, task_2, task_3):
        h1 = self._get_hierarchy_for_task(task_1)
        h2 = self._get_hierarchy_for_task(task_2)
        h3 = self._get_hierarchy_for_task(task_3)

        root = HierarchicalAutomaton(WaterWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u2")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")
        root.add_call_edge("u0", "u1", h1.get_root_automaton_name(), TRUE)
        root.add_call_edge("u1", "u2", h2.get_root_automaton_name(), TRUE)
        root.add_call_edge("u2", "u_acc", h3.get_root_automaton_name(), TRUE)
        return self._build_hierarchy(root, [h1, h2, h3], [])

    def _get_rgb_cym_interavoidance_hierarchy(self, task_name, use_empty):
        if use_empty:
            rgb_ia = self._get_hierarchy_for_task(WaterWorldTasks.REGEBE_INTERAVOIDANCE.value)
            cmy_ia = self._get_hierarchy_for_task(WaterWorldTasks.CEMEYE_INTERAVOIDANCE.value)
        else:
            rgb_ia = self._get_hierarchy_for_task(WaterWorldTasks.RGB_INTERAVOIDANCE.value)
            cmy_ia = self._get_hierarchy_for_task(WaterWorldTasks.CMY_INTERAVOIDANCE.value)

        root = HierarchicalAutomaton(WaterWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u2")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")
        root.add_call_edge("u0", "u1", rgb_ia.get_root_automaton_name(), TRUE)
        root.add_call_edge("u0", "u2", cmy_ia.get_root_automaton_name(), DNFFormula([[f"~{WaterWorldObservables.RED}", f"~{WaterWorldObservables.GREEN}", f"~{WaterWorldObservables.BLUE}"]]))
        root.add_call_edge("u1", "u_acc", cmy_ia.get_root_automaton_name(), TRUE)
        root.add_call_edge("u2", "u_acc", rgb_ia.get_root_automaton_name(), TRUE)
        return self._build_hierarchy(root, [rgb_ia, cmy_ia], [])

    def _get_rgb_cym_avoid_next_two_hierarchy(self, task_name):
        rgb_ant = self._get_hierarchy_for_task(WaterWorldTasks.REGEBE_AVOID_NEXT_TWO.value)
        cmy_ant = self._get_hierarchy_for_task(WaterWorldTasks.CEMEYE_AVOID_NEXT_TWO.value)

        root = HierarchicalAutomaton(WaterWorldEnv.TASK_NAME_TO_AUTOMATON_NAME[task_name])
        root.add_state("u0")
        root.add_state("u1")
        root.add_state("u2")
        root.add_state("u_acc")
        root.set_initial_state("u0")
        root.set_accept_state("u_acc")
        root.add_call_edge("u0", "u1", rgb_ant.get_root_automaton_name(), TRUE)
        root.add_call_edge("u0", "u2", cmy_ant.get_root_automaton_name(), DNFFormula(
            [[f"~{WaterWorldObservables.RED}", f"~{WaterWorldObservables.GREEN}", f"~{WaterWorldObservables.BLUE}"]]))
        root.add_call_edge("u1", "u_acc", cmy_ant.get_root_automaton_name(), TRUE)
        root.add_call_edge("u2", "u_acc", rgb_ant.get_root_automaton_name(), TRUE)
        return self._build_hierarchy(root, [rgb_ant, cmy_ant], [])

    def env_reset(self):
        if self.random_gen is None or not self.random_restart:
            self.random_gen = random.Random(self.seed)

        # Adding the agent
        pos_a = [2 * self.ball_radius + self.random_gen.random() * (self.max_x - 2 * self.ball_radius),
                 2 * self.ball_radius + self.random_gen.random() * (self.max_y - 2 * self.ball_radius)]
        self.agent = BallAgent("A", self.ball_radius, pos_a, [0.0, 0.0], self.agent_vel_delta, self.agent_vel_max)

        # Adding the balls
        self.balls = []

        for color in [
            WaterWorldObservables.RED, WaterWorldObservables.GREEN, WaterWorldObservables.BLUE,
            WaterWorldObservables.CYAN, WaterWorldObservables.MAGENTA, WaterWorldObservables.YELLOW
        ]:
            self._add_balls(color, self.ball_num_per_color)

        if self.avoid_black:
            self._add_balls(WaterWorldObservables.BLACK, self.num_black_balls)

        return self._get_features()

    def _add_balls(self, color, num_balls):
        for _ in range(num_balls):
            pos, vel = self._get_pos_vel_new_ball()
            ball = Ball(color, self.ball_radius, pos, vel)
            self.balls.append(ball)

    def _get_features(self):
        # absolute position and velocity of the agent + relative positions and velocities of the other balls with
        # respect to the agent
        agent, balls = self.agent, self.balls

        if self.use_velocities:
            n_features = 4 + len(balls) * 4
            features = np.zeros(n_features, dtype=np.float32)

            pos_max = np.array([float(self.max_x), float(self.max_y)])
            vel_max = float(self.ball_velocity + self.agent_vel_max)

            features[0:2] = agent.pos / pos_max
            features[2:4] = agent.vel / float(self.agent_vel_max)

            for i in range(len(balls)):
                # if the balls are colliding, they are not included because there is nothing the agent can do about it
                b = balls[i]
                init = 4 * (i + 1)
                features[init:init+2] = (b.pos - agent.pos) / pos_max
                features[init+2:init+4] = (b.vel - agent.vel) / vel_max
        else:
            n_features = 4 + len(balls) * 2
            features = np.zeros(n_features, dtype=np.float)

            pos_max = np.array([float(self.max_x), float(self.max_y)])

            features[0:2] = agent.pos / pos_max
            features[2:4] = agent.vel / float(self.agent_vel_max)

            for i in range(len(balls)):
                b = balls[i]
                init = 2 * i + 4
                features[init:init+2] = (b.pos - agent.pos) / pos_max

        return features

    def _get_feature_size(self):
        base_size = 4  # agent position and velocity
        num_features_per_ball = 4 if self.use_velocities else 2
        num_colored_balls = 6 * self.ball_num_per_color  # There are 6 colors (not counting black)
        num_black_balls = self.num_black_balls if self.avoid_black else 0
        return base_size + (num_colored_balls + num_black_balls) * num_features_per_ball

    def render(self, mode='human'):
        if not self.is_rendering:
            pygame.init()
            pygame.display.set_caption("Water World")
            self.game_display = pygame.display.set_mode((self.max_x, self.max_y))
            self.is_rendering = True

        # printing image
        self.game_display.fill((255, 255, 255))
        for ball in self.balls:
            self._render_ball(self.game_display, ball, 0)
        self._render_ball(self.game_display, self.agent, 3)

        pygame.display.update()

    def _render_ball(self, game_display, ball, thickness):
        pygame.draw.circle(game_display, WaterWorldEnv.RENDERING_COLORS[ball.color],
                           self._get_ball_position(ball, self.max_y), ball.radius, thickness)

    def _get_ball_position(self, ball, max_y):
        return int(round(ball.pos[0])), int(max_y) - int(round(ball.pos[1]))

    def close(self):
        pygame.quit()
        self.is_rendering = False

    def play(self):
        self.reset()
        self.render()

        clock = pygame.time.Clock()

        t_previous = time.time()
        actions = set()

        total_reward = 0.0

        while not self.is_terminal():
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if WaterWorldActions.LEFT in actions and event.key == pygame.K_LEFT:
                        actions.remove(WaterWorldActions.LEFT)
                    elif WaterWorldActions.RIGHT in actions and event.key == pygame.K_RIGHT:
                        actions.remove(WaterWorldActions.RIGHT)
                    elif WaterWorldActions.UP in actions and event.key == pygame.K_UP:
                        actions.remove(WaterWorldActions.UP)
                    elif WaterWorldActions.DOWN in actions and event.key == pygame.K_DOWN:
                        actions.remove(WaterWorldActions.DOWN)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        actions.add(WaterWorldActions.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        actions.add(WaterWorldActions.RIGHT)
                    elif event.key == pygame.K_UP:
                        actions.add(WaterWorldActions.UP)
                    elif event.key == pygame.K_DOWN:
                        actions.add(WaterWorldActions.DOWN)

            t_current = time.time()
            t_delta = (t_current - t_previous)

            # getting the action
            if len(actions) == 0:
                a = WaterWorldActions.NONE
            else:
                a = random.choice(list(actions))

            # executing the action
            _, reward, is_done, _ = self.step(a)  #, t_delta)
            total_reward += reward

            # printing image
            self.render()

            clock.tick(20)

            t_previous = t_current

        print("Game finished. Total reward: %.2f." % total_reward)

        self.close()
