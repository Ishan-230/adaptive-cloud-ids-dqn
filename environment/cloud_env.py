import gym
import numpy as np

class CloudThreatEnv(gym.Env):
    def __init__(self, data):
        super(CloudThreatEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_index = 0
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(data.shape[1] - 1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # 0: Ignore, 1: Alert, 2: Monitor
        self.reset()

    def reset(self):
        self.current_index = 0
        return self._get_obs()

    def _get_obs(self):
        if self.current_index >= len(self.data):
            # Return a dummy observation if out of bounds
            return np.zeros(self.data.shape[1] - 1, dtype=np.float32)
        return self.data.iloc[self.current_index, :-1].values.astype(np.float32)

    def step(self, action):
        if self.current_index >= len(self.data):
            # Episode is done
            return self._get_obs(), 0, True, {}

        label = self.data.iloc[self.current_index, -1]
        reward = 0
        if action == 1 and label == 1:
            reward = 1
        elif action == 1 and label == 0:
            reward = -1
        elif action == 0 and label == 1:
            reward = -2
        elif action == 2:
            reward = 0.5

        self.current_index += 1
        done = self.current_index >= len(self.data)

        obs = self._get_obs()
        return obs, reward, done, {}
