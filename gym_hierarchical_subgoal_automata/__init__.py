from gym.envs.registration import register
from gym_hierarchical_subgoal_automata.envs.craftworld.craftworld_env import CraftWorldTasks
from gym_hierarchical_subgoal_automata.envs.waterworld.waterworld_env import WaterWorldObservables, WaterWorldTasks


def _get_cw_id(task_name):
    return "".join([x.title() for x in task_name.split("-")])


for task_name in CraftWorldTasks.list():
    register(
        id=f'CraftWorld{_get_cw_id(task_name)}-v0',
        entry_point='gym_hierarchical_subgoal_automata.envs.craftworld:CraftWorldEnv',
        kwargs={'task_name': task_name}
    )


# One and two subgoals sequences
def _n_subgoal_id(task_name):
    return "".join([x.upper() for x in task_name.split("-")])


for task in [
    WaterWorldTasks.R, WaterWorldTasks.G, WaterWorldTasks.B, WaterWorldTasks.C, WaterWorldTasks.M, WaterWorldTasks.Y,
    WaterWorldTasks.RG, WaterWorldTasks.BC, WaterWorldTasks.MY,  WaterWorldTasks.RE, WaterWorldTasks.GE,
    WaterWorldTasks.BE, WaterWorldTasks.CE, WaterWorldTasks.YE, WaterWorldTasks.ME, WaterWorldTasks.RGB,
    WaterWorldTasks.CMY
]:
    register(
        id=f'WaterWorld{_n_subgoal_id(task.value)}-v0',
        entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
        kwargs={'task_name': task.value}
    )

# Diamond tasks
for task in [
    WaterWorldTasks.RG_BC, WaterWorldTasks.BC_MY, WaterWorldTasks.RG_MY, WaterWorldTasks.RGB_CMY, WaterWorldTasks.RG_BC_MY
]:
    register(
        id=f'WaterWorld{"And".join([_n_subgoal_id(s[1:-1]) for s in task.value.split("&")])}-v0',
        entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
        kwargs={'task_name': task.value}
    )

for task in [
    WaterWorldTasks.R_WO_G, WaterWorldTasks.R_WO_GB, WaterWorldTasks.R_WO_GBC, WaterWorldTasks.R_WO_GBCY,
    WaterWorldTasks.R_WO_GBCYM, WaterWorldTasks.G_WO_B, WaterWorldTasks.G_WO_BC, WaterWorldTasks.G_WO_BCY,
    WaterWorldTasks.G_WO_BCYM, WaterWorldTasks.G_WO_BCYMR, WaterWorldTasks.B_WO_C, WaterWorldTasks.B_WO_CY,
    WaterWorldTasks.B_WO_CYM, WaterWorldTasks.B_WO_CYMR, WaterWorldTasks.B_WO_CYMRG, WaterWorldTasks.C_WO_Y,
    WaterWorldTasks.C_WO_YM, WaterWorldTasks.C_WO_YMR, WaterWorldTasks.C_WO_YMRG, WaterWorldTasks.C_WO_YMRGB,
    WaterWorldTasks.Y_WO_M, WaterWorldTasks.Y_WO_MR, WaterWorldTasks.Y_WO_MRG, WaterWorldTasks.Y_WO_MRGB,
    WaterWorldTasks.Y_WO_MRGBC, WaterWorldTasks.M_WO_R, WaterWorldTasks.M_WO_RG, WaterWorldTasks.M_WO_RGB,
    WaterWorldTasks.M_WO_RGBC, WaterWorldTasks.M_WO_RGBCY, WaterWorldTasks.G_WO_BR, WaterWorldTasks.B_WO_R,
    WaterWorldTasks.B_WO_RG, WaterWorldTasks.Y_WO_MC, WaterWorldTasks.M_WO_C, WaterWorldTasks.M_WO_CY
]:
    target, to_avoid = task.value.split("\\")
    register(
        id=f'WaterWorld{target.upper()}Without{to_avoid.upper()}-v0',
        entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
        kwargs={'task_name': task.value}
    )

# Base avoidance tasks
for task in [
    WaterWorldTasks.RE_WO_G, WaterWorldTasks.RE_WO_GB, WaterWorldTasks.GE_WO_B, WaterWorldTasks.GE_WO_BR,
    WaterWorldTasks.BE_WO_R, WaterWorldTasks.BE_WO_RG, WaterWorldTasks.CE_WO_M, WaterWorldTasks.CE_WO_MY,
    WaterWorldTasks.ME_WO_Y, WaterWorldTasks.ME_WO_YC, WaterWorldTasks.YE_WO_C, WaterWorldTasks.YE_WO_CM,

    WaterWorldTasks.GE_WO_BC, WaterWorldTasks.BE_WO_C, WaterWorldTasks.BE_WO_CM, WaterWorldTasks.ME_WO_YR,
    WaterWorldTasks.YE_WO_R, WaterWorldTasks.YE_WO_RG
]:
    target, to_avoid = task.value.split("-")[0][1:-1].split("\\")
    register(
        id=f'WaterWorld{target.upper()}{WaterWorldObservables.EMPTY.upper()}Without{to_avoid.upper()}-v0',
        entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
        kwargs={'task_name': task.value}
    )

# Avoidance sequences
for task in [WaterWorldTasks.RGB_FULL_STRICT, WaterWorldTasks.CMY_FULL_STRICT]:
    register(
        id=f'WaterWorld{"".join([s[1] for s in task.value.split("-")]).upper()}FullStrict-v0',
        entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
        kwargs={'task_name': task.value}
    )

for task in [WaterWorldTasks.RGB_INTERAVOIDANCE, WaterWorldTasks.CMY_INTERAVOIDANCE]:
    register(
        id=f'WaterWorld{"".join([s.strip("(")[0] for s in task.value.split(")-(")]).upper()}Interavoidance-v0',
        entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
        kwargs={'task_name': task.value}
    )

for task in [WaterWorldTasks.REGEBE_INTERAVOIDANCE, WaterWorldTasks.CEMEYE_INTERAVOIDANCE]:
    register(
        id=f'WaterWorld{"".join([s.strip("(")[0] for s in task.value.split(")-(")]).upper()}EmptyInteravoidance-v0',
        entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
        kwargs={'task_name': task.value}
    )

for task in [WaterWorldTasks.REGEBE_AVOID_NEXT_TWO, WaterWorldTasks.CEMEYE_AVOID_NEXT_TWO]:
    register(
        id=f'WaterWorld{"".join([s.strip("(")[0] for s in task.value.split(")-(")]).upper()}EmptyAvoidNextTwo-v0',
        entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
        kwargs={'task_name': task.value}
    )

register(
    id='WaterWorldRGBAndCMYInteravoidance-v0',
    entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
    kwargs={'task_name': WaterWorldTasks.RGB_CMY_INTERAVOIDANCE.value}
)

register(
    id='WaterWorldRGBAndCMYEmptyInteravoidance-v0',
    entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
    kwargs={'task_name': WaterWorldTasks.REGEBE_CEMEYE_INTERAVOIDANCE.value}
)

register(
    id='WaterWorldRGBAndCMYEmptyAvoidNextTwo-v0',
    entry_point='gym_hierarchical_subgoal_automata.envs.waterworld:WaterWorldEnv',
    kwargs={'task_name': WaterWorldTasks.REGEBE_CEMEYE_AVOID_NEXT_TWO.value}
)
