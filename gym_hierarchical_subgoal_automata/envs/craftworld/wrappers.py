import gym
from gym_minigrid.minigrid import COLORS, OBJECT_TO_IDX, STATE_TO_IDX, DIR_TO_VEC
import numpy as np


class TabularWrapper(gym.ObservationWrapper):
    """
    Returns tabular observations: an integer whose range depends on the number of directions in the environment (4), and
    the width and height of the grid.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Discrete(self.env.height * self.env.width * self.env.NUM_DIRECTIONS)

    def observation(self, observation):
        i, j = self.env.agent_pos
        return self.env.agent_dir + self.env.NUM_DIRECTIONS * j + self.env.NUM_DIRECTIONS * self.env.height * i


class OneHotWrapper(gym.ObservationWrapper):
    """
    Returns one-hot observations. The length of the vector is height * width * number of orientations (4).
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.MultiBinary(self.env.height * self.env.width * 4)

    def observation(self, observation):
        agent_pos, agent_dir = self.env.agent_pos, self.env.agent_dir

        obs = np.zeros((self.env.height, self.env.width, self.env.NUM_DIRECTIONS))
        obs[agent_pos[1], agent_pos[0], agent_dir] = 1
        final_obs = obs.flatten()
        return final_obs


class FullyObsTransform(gym.core.Wrapper):
    """
    In order to reuse the architecture from "Prioritized Level Replay", I partially copied this wrapper which returns
    the observations with a slightly different shape (just changes the order of the dimensions). Besides, the
    observations can be also scaled.
    """
    def __init__(self, env, scale_obs_mode, remove_colors):
        super(FullyObsTransform, self).__init__(env)
        m, n, c = env.observation_space.shape
        if remove_colors:  # the color channel is removed
            c -= 1
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [c, m, n],
            dtype=self.observation_space.dtype)
        self.scale_obs_mode = scale_obs_mode
        self.remove_colors = remove_colors

    def reset(self):
        obs = self.env.reset()
        return self._transform(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._transform(obs), reward, done, info

    def _transform(self, obs):
        if obs.shape[1] != 3:
            obs = obs.transpose(2, 0, 1)
        if self.scale_obs_mode == "zero_one" or self.scale_obs_mode == "minus_one_one":
            # For the third matrix, we normalize by whatever more values are specified: directions of the agent, or
            # states of the objects (closed, locked, ...).
            scale_factor = np.array([
                [1 / (len(OBJECT_TO_IDX) - 1)],
                [1 / (len(COLORS) - 1)],
                [1 / (max(len(DIR_TO_VEC), len(STATE_TO_IDX)) - 1)]
            ], dtype=np.float32)
            obs = obs * scale_factor[:, None]
            if self.scale_obs_mode == "minus_one_one":
                obs = 2 * obs - 1
        if self.remove_colors:
            nc_obs = np.zeros(self.observation_space.shape)
            nc_obs[0], nc_obs[1] = obs[0], obs[2]  # copy object id and object state channels
            return nc_obs
        return obs


class RunningMeanStd():
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizedEnv(gym.core.Wrapper):
    """
    Taken taken from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py, which in turn is
    a non-vectorized form taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py.
    This is used in the paper "Prioritized Level Replay", which also uses GymMinigrid in the full observable setting.
    """
    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        infos['real_reward'] = rews
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret = self.ret * (1-float(dones))
        return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(())
        obs = self.env.reset()
        return self._obfilt(obs)
