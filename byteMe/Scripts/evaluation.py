import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from train import load_and_preprocess_data, LSTMModel, TransformerModel, MLPAdapter

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
plot_dir = os.path.join(base_dir, 'Visualizations')
models_dir = os.path.join(base_dir, 'Models')
data_dir = os.path.join(base_dir, 'Data')

def evaluate_main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Evaluating models on backend: {device}", flush=True)
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory missing at {models_dir}")
        return

    _, _, X_test, Y_test, Bsl_test, test_idx, valid_df, seq_len, _, horizons, _ = load_and_preprocess_data()
    n_test, t_len, num_f = X_test.shape
    
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    X_test_flat = X_test.reshape(-1, num_f)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_test, t_len, num_f)

    # Dictionary to hold predictions
    model_preds = {}

    # Load and evaluate Scikit-Learn / XGBoost models
    print("\nEvaluating RF...", flush=True)
    rf = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
    model_preds['Multi-RF'] = rf.predict(X_test[:, -1, :])
    
    print("Evaluating XGBoost...", flush=True)
    xgb_model = joblib.load(os.path.join(models_dir, 'xgb_model.pkl'))
    model_preds['Multi-XGB'] = xgb_model.predict(X_test[:, -1, :])

    # Loader for PyTorch
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(Y_test))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    def evaluate_torch(name, model):
        print(f"Evaluating {name}...", flush=True)
        model_device = torch.device('cpu') if (name == 'LSTM' and device.type == 'mps') else device
        model.load_state_dict(torch.load(os.path.join(models_dir, f"{name.lower()}_model.pt"), map_location=model_device))
        model.to(model_device)
        model.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(model_device)
                preds.append(model(bx).cpu().numpy())
        return np.vstack(preds)
    
    model_preds['MLP'] = evaluate_torch("MLP", MLPAdapter(num_f, out_dim=len(horizons)))
    model_preds['LSTM'] = evaluate_torch("LSTM", LSTMModel(num_f, out_dim=len(horizons)))
    model_preds['Transformer'] = evaluate_torch("Transformer", TransformerModel(num_f, out_dim=len(horizons)))

    # Persistent baseline: predict 0 difference, meaning value stays the same
    model_preds['Persistent'] = np.zeros((n_test, len(horizons)))

    results = []
    print("\n--- Model Performance Summary ---", flush=True)
    for name, preds_diff in model_preds.items():
        for i, h in enumerate(horizons):
            actual_h = Bsl_test + Y_test[:, i]
            recon_h = Bsl_test + preds_diff[:, i]
            mse = mean_squared_error(actual_h, recon_h)
            r2 = r2_score(actual_h, recon_h)
            print(f"[{name}] Horizon +{h}m -> MSE: {mse:.4f} | R2: {r2:.4f}")
            results.append({'Model': name, 'Horizon': h, 'MSE': mse, 'R2': r2})

    res_df = pd.DataFrame(results)
    
    # Save the predicted metric data directly into CSV 
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'model_performance_metrics.csv')
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV to {csv_path}", flush=True)

    # Save the predicted values from each model and horizon
    preds_df_dict = {'Test_Idx': test_idx, 'Baseline': Bsl_test}
    for i, h in enumerate(horizons):
        preds_df_dict[f'Actual_+{h}m'] = Bsl_test + Y_test[:, i]
        for name in model_preds.keys():
            preds_df_dict[f'{name}_+{h}m'] = Bsl_test + model_preds[name][:, i]
    
    preds_df = pd.DataFrame(preds_df_dict)
    preds_csv_path = os.path.join(data_dir, 'ml_results.csv')
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"Saved predictions CSV to {preds_csv_path}", flush=True)

    print("\nTabular Pivot:")
    print(res_df.pivot(index='Model', columns='Horizon', values=['MSE', 'R2']))

    # Visualizations
    colors = {'Multi-RF': 'blue', 'Multi-XGB': 'green', 'LSTM': 'red', 'Transformer': 'orange', 'MLP': 'purple', 'Persistent': 'gray'}
    
    print("\nGenerating consolidated visualizations...", flush=True)
    for i, h in enumerate(horizons):
        plt.figure(figsize=(14, 6))
        actual_h = Bsl_test + Y_test[:, i]
        plt.plot(range(len(actual_h)), actual_h, label=f'Actual TN +{h}m', color='black', alpha=0.5)

        for name in model_preds.keys():
            recon_h = Bsl_test + model_preds[name][:, i]
            plt.plot(range(len(recon_h)), recon_h, label=f'{name}', color=colors[name], alpha=0.6)

        plt.title(f'Consolidated Model Comparisons (+{h}min Extrapolation)')
        plt.xlabel('Test Time Steps')
        plt.ylabel('TN Temp Rise (°C)')
        plt.legend(loc='lower left', ncol=3)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/consolidated_comparison_horizon_{h}m.png")
        plt.close()
        
        # Absolute error trace
        plt.figure(figsize=(14, 6))
        for name in model_preds.keys():
            recon_h = Bsl_test + model_preds[name][:, i]
            error_h = np.abs(recon_h - actual_h)
            plt.plot(range(len(error_h)), error_h, label=f'{name}', color=colors[name], alpha=0.7, linewidth=1)

        plt.title(f'Absolute Error Test Trace (+{h}min Extrapolation)')
        plt.xlabel('Test Time Steps')
        plt.ylabel('Absolute Error |Pred - Actual| (°C)')
        plt.legend(loc='upper right', ncol=3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"error_trace_horizon_{h}m.png"))
        plt.close()
        
        # Scatter plot truth vs predictions
        plt.figure(figsize=(10, 10))
        min_v = np.min(actual_h)
        max_v = np.max(actual_h)
        for name in model_preds.keys():
            recon_h = Bsl_test + model_preds[name][:, i]
            min_v = min(min_v, np.min(recon_h))
            max_v = max(max_v, np.max(recon_h))
            plt.scatter(actual_h, recon_h, label=f'{name}', color=colors.get(name, 'black'), alpha=0.3, s=10)
        
        plt.plot([min_v, max_v], [min_v, max_v], 'k--', label='Ideal', alpha=0.7)
        plt.title(f'Prediction vs Truth Scatter (+{h}min Extrapolation)')
        plt.xlabel('Actual TN Temp Rise (°C)')
        plt.ylabel('Predicted TN Temp Rise (°C)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"scatter_horizon_{h}m.png"))
        plt.close()

        
    print(f"All evaluation plots successfully saved to: {plot_dir}")

if __name__ == '__main__':
    evaluate_main()
