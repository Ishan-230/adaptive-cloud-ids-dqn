from utils.preprocess import load_and_preprocess
from environment.cloud_env import CloudThreatEnv
from models.dqn_agent import train_dqn

if __name__ == '__main__':
    data_path = "data/simulated_logs.csv"
    df = load_and_preprocess(data_path)
    env = CloudThreatEnv(df)
    model = train_dqn(env)
    model.save("results/final_model")
    print("✅ Model trained and saved at results/final_model.zip")