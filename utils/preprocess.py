import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=["timestamp", "user_id"], inplace=True)  # Drop non-numeric features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['label'] = y.values
    return df_scaled