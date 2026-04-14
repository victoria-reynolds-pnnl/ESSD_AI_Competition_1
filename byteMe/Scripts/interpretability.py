import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import PartialDependenceDisplay

try:
    import shap
except ImportError:
    import subprocess
    print("Installing SHAP...")
    subprocess.call(['pip', 'install', 'shap'])
    import shap

def load_environment():
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    from train import load_and_preprocess_data
    return load_and_preprocess_data()

def generate_shap_plots(xgb_model, X_test_2d, feature_names, horizons, out_dir):
    print("Generating SHAP plots...", flush=True)
    for i, h in enumerate(horizons):
        # We generate SHAP for 15m and 240m to isolate short vs long forecasting explanations
        if h not in [15, 240]:
            continue
        try:
            model = xgb_model.estimators_[i]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_2d[:2000]) # Cap eval to 2000 for visibility

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_2d[:2000], feature_names=feature_names, show=False)
            plt.title(f'SHAP Beeswarm: Multi-XGBoost (+{h}m Horizon)', pad=15, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'SHAP_Analysis_XGB_Horizon_{h}m.png'))
            plt.close()
        except Exception as e:
            print(f"SHAP generation failed for horizon {h}: {e}")

def generate_feature_importance(xgb_model, feature_names, horizons, out_dir):
    print("Generating Shifting Feature Importance Matrix...", flush=True)
    importances = {}
    for i, h in enumerate(horizons):
        importances[f'+{h}m'] = xgb_model.estimators_[i].feature_importances_

    df_imp = pd.DataFrame(importances, index=feature_names)
    df_imp['Mean'] = df_imp.mean(axis=1)
    df_imp = df_imp.sort_values(by='Mean', ascending=True).tail(10) # Limit to Top 10 Features
    df_imp.drop('Mean', axis=1, inplace=True)

    df_imp.plot(kind='barh', figsize=(11, 7), width=0.8, colormap='Spectral')
    plt.title('Top 10 Feature Importances Shifting Across Horizons (XGBoost)', fontsize=14, fontweight='bold')
    plt.xlabel('F-Score (Relative Importance)', fontsize=12)
    plt.ylabel('Engineered Features', fontsize=12)
    plt.legend(title='Prediction Horizon', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Interpretability_Feature_Importance_Shifting.png'))
    plt.close()

def generate_pdp(xgb_model, X_test_2d, feature_names, out_dir):
    print("Generating Partial Dependence Plots...", flush=True)
    try:
        model = xgb_model.estimators_[0] 
        df_x = pd.DataFrame(X_test_2d, columns=feature_names)
        
        features_to_plot = []
        for f in ['cumulative_heat_input', 'net_flow_rolling_6h', 'TC_INT_delta']:
            if f in feature_names:
                features_to_plot.append(f)
                
        if not features_to_plot:
            features_to_plot = [0, 1] 

        fig, ax = plt.subplots(figsize=(14, 6))
        PartialDependenceDisplay.from_estimator(model, df_x, features_to_plot, ax=ax, grid_resolution=50)
        plt.suptitle('Partial Dependence: Physical Saturation Effects (+15m Horizon)', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'Interpretability_Partial_Dependence.png'))
        plt.close()
    except Exception as e:
        print(f"PDP generation failed: {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    out_dir = os.path.join(base_dir, 'Visualizations')
    models_dir = os.path.join(base_dir, 'Models')
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Recover Core Baseline Plot R2
    data_dir = os.path.join(base_dir, 'Data')
    csv_path = os.path.join(data_dir, 'model_performance_metrics.csv')
    if os.path.exists(csv_path):
        df_metrics = pd.read_csv(csv_path)
        df_metrics.rename(columns={'Horizon': 'Horizon_Min'}, inplace=True)
        colors = {'Multi-RF': 'tab:blue', 'Multi-XGB': 'tab:green', 'MLP': 'tab:purple', 'LSTM': 'tab:red', 'Transformer': 'tab:orange', 'Persistent': 'tab:gray'}
        
        plt.figure(figsize=(11, 5))
        for model in df_metrics['Model'].unique():
            subset = df_metrics[df_metrics['Model'] == model]
            plt.plot(subset['Horizon_Min'], subset['R2'], marker='o', label=model, linewidth=2.5, color=colors.get(model, 'black'))
        plt.xscale('log')
        plt.xticks([15, 60, 240, 1440], ['15m', '1h', '4h', '24h'])
        plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
        plt.title('Predictive Accuracy (R²) across Time Horizons', fontsize=14, fontweight='bold')
        plt.ylabel('R² Score', fontsize=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'Interpretability_R2_vs_Horizon.png'))
        plt.close()
    
    # 2. Advanced Interpretability Engineering
    print("Loading models and test inferences for Advanced Interpretability...", flush=True)
    _, _, X_test, Y_test, Bsl_test, _, _, _, extended_features, horizons, _ = load_environment()
    
    xgb_path = os.path.join(models_dir, 'xgb_model.pkl')
    if not os.path.exists(xgb_path):
        print("XGBoost model missing. Run training pipeline.")
        return
        
    xgb_model = joblib.load(xgb_path)
    X_test_2d = X_test[:, -1, :] 
    
    # Pipeline Execute
    generate_shap_plots(xgb_model, X_test_2d, extended_features, horizons, out_dir)
    generate_feature_importance(xgb_model, extended_features, horizons, out_dir)
    generate_pdp(xgb_model, X_test_2d, extended_features, out_dir)
    
if __name__ == '__main__':
    main()
