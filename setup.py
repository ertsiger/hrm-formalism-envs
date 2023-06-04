from setuptools import setup

setup(
      name='gym_hierarchical_subgoal_automata',
      version='0.0.1',
      install_requires=[
            "gym~=0.15.3",
            "matplotlib==3.4.3",
            "gym_minigrid @ git+https://github.com/ertsiger/hrm-minigrid.git",
            "numpy==1.21.3",
            "pygame==1.9.6"
      ]
)
