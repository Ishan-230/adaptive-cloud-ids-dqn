import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_simulated_logs(num_entries=1000, attack_ratio=0.3, output_path="data/simulated_logs.csv"):
    logs = []
    start_time = datetime.now()

    for i in range(num_entries):
        timestamp = (start_time + timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S')
        user_id = f"user{random.randint(1, 50):03}"
        login_attempts = np.random.poisson(1)
        cpu_usage = np.random.uniform(5, 95)
        port_scans = np.random.randint(0, 5)
        file_changes = np.random.randint(0, 10)

        # Decide if this row is malicious
        if random.random() < attack_ratio:
            label = 1  # Attack
            login_attempts = np.random.randint(5, 15)
            cpu_usage = np.random.uniform(80, 100)
            port_scans = np.random.randint(5, 15)
            file_changes = np.random.randint(5, 20)
        else:
            label = 0  # Normal

        logs.append([timestamp, user_id, login_attempts, cpu_usage, port_scans, file_changes, label])

    df = pd.DataFrame(logs, columns=[
        "timestamp", "user_id", "login_attempts", "cpu_usage",
        "port_scans", "file_changes", "label"
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Generated simulated logs: {output_path} with {num_entries} entries")

if __name__ == "__main__":
    generate_simulated_logs()
