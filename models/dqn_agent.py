from stable_baselines3 import DQN

def train_dqn(env):
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0005, buffer_size=10000)
    model.learn(total_timesteps=10000)
    return model