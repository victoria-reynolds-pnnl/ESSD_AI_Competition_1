import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
DATA_DIR = os.path.join(base_dir, 'Data')
MODELS_DIR = os.path.join(base_dir, 'Models')

def load_and_preprocess_data():
    file_path = os.path.join(DATA_DIR, 'week2_data.csv')
    splits_path = os.path.join(DATA_DIR, 'splits_indexed.csv')
    print("Loading data...", flush=True)
    df = pd.read_csv(file_path, parse_dates=['timestamp']).sort_values('timestamp')
    if os.path.exists(splits_path) and 'split' not in df.columns:
        splits_df = pd.read_csv(splits_path)
        df['split'] = splits_df['split']


    base_features = [
        'cumulative_heat_input', 'elapsed_injection_min', 'net_flow_rolling_6h', 
        'TC_INT_delta', 'T_gradient_INT_TC', 'days_since_injection',
        'hour_sin', 'hour_cos', 'delta_T_above_T0_TN'
    ]
    target_col = 'delta_T_above_T0_TN'
    horizons = [15, 60, 240, 1440]

    for lag in [15, 30, 60]:
        df[f'target_lag_{lag}'] = df[target_col].shift(lag)
        df[f'flow_lag_{lag}'] = df['net_flow_rolling_6h'].shift(lag)
    
    extended_features = base_features + [f'target_lag_{l}' for l in [15, 30, 60]] + [f'flow_lag_{l}' for l in [15, 30, 60]]

    target_diff_cols = []
    for h in horizons:
        col = f'target_diff_{h}'
        df[col] = df[target_col].shift(-h) - df[target_col]
        target_diff_cols.append(col)

    valid_df = df.dropna(subset=extended_features + target_diff_cols).reset_index(drop=True)

    seq_len = 30
    feature_matrix = valid_df[extended_features].values
    target_matrix = valid_df[target_diff_cols].values
    split_roles = valid_df['split'].values
    baseline_matrix = valid_df[target_col].values

    def create_sequences(data, targets, splits, baselines, seq_len):
        xs, ys, sps, bs = [], [], [], []
        # Optimization constraint: we need tabular features for RF/XGB and sequential for DL.
        # We will use the last timestep's tabular features for tree models.
        for i in range(len(data) - seq_len):
            xs.append(data[i:(i+seq_len)])
            ys.append(targets[i+seq_len-1])
            sps.append(splits[i+seq_len-1])
            bs.append(baselines[i+seq_len-1])
        return np.array(xs), np.array(ys), np.array(sps), np.array(bs)

    X, Y, Spl, Bsl = create_sequences(feature_matrix, target_matrix, split_roles, baseline_matrix, seq_len)

    train_idx = np.where(Spl == 'train')[0]
    test_idx = np.where(Spl == 'test')[0]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test, Bsl_test = X[test_idx], Y[test_idx], Bsl[test_idx]
    
    return X_train, Y_train, X_test, Y_test, Bsl_test, test_idx, valid_df, seq_len, extended_features, horizons, target_diff_cols

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, out_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        last_out = lstm_out[:, -1, :] 
        return self.fc(last_out)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, out_dim=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, out_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        last_out = out[:, -1, :] 
        return self.fc(last_out)

class MLPAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    def forward(self, x):
        x_flat = x[:, -1, :] # standard tabular uses last timestep
        return self.net(x_flat)

def train_main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Hardware Acceleration Mode: Using device {device} ---", flush=True)

    X_train, Y_train, _, _, _, _, _, _, _, horizons, _ = load_and_preprocess_data()
    n_train, t_len, num_f = X_train.shape

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, num_f)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_train, t_len, num_f)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("StandardScaler saved.", flush=True)

    print("\n[1/5] Training Multi-RF...", flush=True)
    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1, max_depth=8)
    rf.fit(X_train[:, -1, :], Y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR, 'rf_model.pkl'))

    print("\n[2/5] Training Multi-XGBoost...", flush=True)
    xgb_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.01, random_state=42, n_jobs=1))
    xgb_model.fit(X_train[:, -1, :], Y_train)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.pkl'))

    batch_size = 512
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(Y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def train_torch_model(name, model, lr=1e-3, max_epochs=50, patience=4, tolerance=1e-4):
        model_device = torch.device('cpu') if (name == 'LSTM' and device.type == 'mps') else device
        print(f"\nTraining {name} on {model_device}...", flush=True)
        model.to(model_device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for ep in range(max_epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(model_device), by.to(model_device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"[{name}] Epoch {ep+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}", flush=True)
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name.lower()}_model.pt"))

    print("\n[3/5] PyTorch MLP Equivalent...", flush=True)
    train_torch_model("MLP", MLPAdapter(num_f, out_dim=len(horizons)), lr=0.001, max_epochs=50)

    print("\n[4/5] PyTorch Deep Sequence: LSTM...", flush=True)
    torch.manual_seed(42)
    train_torch_model("LSTM", LSTMModel(num_f, out_dim=len(horizons)), lr=0.0005, max_epochs=3)
    
    print("\n[5/5] PyTorch Deep Sequence: Transformer...", flush=True)
    train_torch_model("Transformer", TransformerModel(num_f, out_dim=len(horizons)), lr=0.001, max_epochs=50)

    print("\nAll models trained and exported to Models directory.", flush=True)

if __name__ == '__main__':
    train_main()
