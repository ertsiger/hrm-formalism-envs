import gym
from gym_hierarchical_subgoal_automata.envs.craftworld.craftworld_env import CraftWorldObservables


if __name__ == "__main__":
    # Getting an environment (the grid type is irrelevant for these examples)
    env = gym.make(
        "gym_hierarchical_subgoal_automata:CraftWorldBook-v0",
        params={
            "environment_seed": 0, "grid_params": {"grid_type": "open_plan", "width": 7, "height": 7}
        }
    )

    # Get hierarchy from the defined environment (it consists of more than one automaton in this case)
    hierarchy = env.get_hierarchy()

    # Print states as trace is traversed
    trace = [
        {CraftWorldObservables.SUGARCANE}, {CraftWorldObservables.WORKBENCH}, {}, {CraftWorldObservables.RABBIT},
        {CraftWorldObservables.WORKBENCH}, {CraftWorldObservables.TABLE}
    ]

    # Print traversal
    hierarchy_state = hierarchy.get_initial_state()
    for observation in trace:
        print(hierarchy_state)
        hierarchy_state = hierarchy.get_next_hierarchy_state(hierarchy_state, observation)
    print(hierarchy_state)

    # Print the formulas in the hierarchy
    hierarchy_subgoals = set()
    hierarchy.get_subgoals(hierarchy_subgoals)
    print("Conjunctive subgoals in the hierarchy:", hierarchy_subgoals)

    # Print the local subgoals
    print("Subgoals in the root:", hierarchy.get_root_automaton().get_automaton_subgoals())

    # Plot the root automaton
    for automaton_name in hierarchy.get_automata_names():
        hierarchy.get_automaton(automaton_name).plot(".", f"{automaton_name}.png")
