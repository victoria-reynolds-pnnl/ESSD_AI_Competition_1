"""
Extract GNN edge weight insights and generate text summaries for LLM classification.

Flow:
1. Load edge_weights_over_time.npy (from Week 3)
2. Load ftes_scaled_for_GNN.csv (sensor data)
3. Extract edge weight statistics per window
4. Generate mixed technical+domain narrative text
5. Assign risk confidence based on weight patterns
6. Output CSV compatible with Week 4 LLM notebooks
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

WEEK3_DIR = Path('..') / 'Week3'
EDGE_WEIGHTS_PATH = WEEK3_DIR / 'edge_weights_over_time.npy'
SENSOR_DATA_PATH = WEEK3_DIR / 'ftes_scaled_for_GNN.csv'

OUTPUT_DIR = Path('.') / 'gnn_generated_data'
OUTPUT_CSV = OUTPUT_DIR / 'gnn_insights_for_llm.csv'

MAX_RECORDS = 1500

# Node names (10 nodes, 2 features each from GNN)
NODE_NAMES = [
    'TL (Transport Line)',
    'TN (Transport Node)',
    'TC (Central Tank)',
    'TU (Upper Tank)',
    'TS (Storage)'
]
NODE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Sequential connections
    (0, 2), (1, 3), (2, 4),           # Cross connections
    (0, 3), (1, 4),                   # Long-range
    (0, 4)                            # Full span
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_weight(w: float) -> float:
    """Normalize edge weight to [0, 1] for interpretation."""
    return max(0.0, min(1.0, (w + 1.0) / 2.0))  # Map [-1, 1] to [0, 1]


def extract_window_statistics(edge_weights: np.ndarray) -> dict:
    """
    Extract statistics from edge weights for one time window.
    
    Args:
        edge_weights: shape [num_edges, 1] or [num_edges]
    
    Returns:
        dict with mean, std, max, min, active_edges, etc.
    """
    weights_flat = edge_weights.flatten()
    
    return {
        'mean': float(np.mean(weights_flat)),
        'std': float(np.std(weights_flat)),
        'max': float(np.max(weights_flat)),
        'min': float(np.min(weights_flat)),
        'median': float(np.median(weights_flat)),
        'active_edges': int(np.sum(weights_flat > 0.5)),
        'total_edges': len(weights_flat),
        'variance': float(np.var(weights_flat)),
        'range': float(np.max(weights_flat) - np.min(weights_flat)),
    }


def classify_connectivity_pattern(stats: dict) -> Tuple[str, float]:
    """
    Classify network pattern and return pattern name + confidence.
    
    Returns:
        (pattern_name, confidence_score)
    """
    mean_w = stats['mean']
    std_w = stats['std']
    active_ratio = stats['active_edges'] / stats['total_edges']
    
    # High connectivity: strong average weight, low variance
    if mean_w > 0.4 and std_w < 0.3:
        return ('high_stability', 0.92)
    
    # Moderate but consistent
    if 0.2 < mean_w <= 0.4 and std_w < 0.35:
        return ('moderate_connectivity', 0.85)
    
    # Sparse or weak
    if mean_w <= 0.2 or active_ratio < 0.4:
        return ('sparse_network', 0.78)
    
    # Volatile: high variance despite decent mean
    if std_w > 0.45:
        return ('high_volatility', 0.80)
    
    # Mixed patterns
    if 0.3 < mean_w <= 0.5 and 0.3 < std_w <= 0.45:
        return ('mixed_dynamics', 0.82)
    
    return ('unknown_pattern', 0.60)


def generate_narrative_text(
    sample_idx: int,
    stats: dict,
    pattern: str,
    top_edges: List[Tuple[int, int, float]],
    sensor_snapshot: dict = None
) -> str:
    """
    Generate mixed technical + domain narrative text.
    
    Args:
        sample_idx: window/record index
        stats: connectivity statistics
        pattern: pattern classification
        top_edges: list of (node_i, node_j, weight) for strongest edges
        sensor_snapshot: optional dict with sensor values at this window
    
    Returns:
        narrative text string
    """
    lines = []
    
    # Technical summary
    lines.append(
        f"Network snapshot {sample_idx}: Edge weight analysis "
        f"(mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
        f"active={stats['active_edges']}/{stats['total_edges']})."
    )
    
    # Pattern interpretation
    if pattern == 'high_stability':
        lines.append(
            "System exhibits stable, coherent connectivity across all nodes. "
            "Strong cross-node correlation suggests synchronized operational state."
        )
    elif pattern == 'moderate_connectivity':
        lines.append(
            "Moderate node coupling detected with selective strong edges. "
            "Network maintains partial coordination with some independent regions."
        )
    elif pattern == 'sparse_network':
        lines.append(
            "Network is sparsely connected with weak average coupling. "
            "Nodes operate quasi-independently with limited information flow."
        )
    elif pattern == 'high_volatility':
        lines.append(
            "High variance in edge weights indicates unstable or transitional network state. "
            "Some node pairs show strong coupling while others remain disconnected."
        )
    elif pattern == 'mixed_dynamics':
        lines.append(
            "Network exhibits mixed dynamics with moderate mean coupling but significant variance. "
            "Suggests partial synchronization with local clustering."
        )
    else:
        lines.append("Network pattern is indeterminate from edge statistics.")
    
    # Top edges
    if top_edges:
        top_edge_descriptions = []
        for i, j, w in top_edges[:3]:
            if i < len(NODE_NAMES) and j < len(NODE_NAMES):
                node_i = NODE_NAMES[i // 2] if i < 10 else f"Node{i}"
                node_j = NODE_NAMES[j // 2] if j < 10 else f"Node{j}"
            else:
                node_i, node_j = f"Node{i}", f"Node{j}"
            
            strength = "very strong" if w > 0.7 else "strong" if w > 0.5 else "moderate"
            top_edge_descriptions.append(f"{node_i}–{node_j} ({strength}, w={w:.2f})")
        
        lines.append(
            f"Strongest edges: {', '.join(top_edge_descriptions)}. "
            "These represent primary signal transmission pathways."
        )
    
    # Optional sensor context
    if sensor_snapshot:
        lines.append(
            f"Associated sensor readings: net_flow={sensor_snapshot.get('net_flow', 'N/A'):.2f}, "
            f"injection_pressure={sensor_snapshot.get('injection_pressure', 'N/A'):.2f}."
        )
    
    # Variability assessment
    if stats['range'] > 1.5:
        lines.append(
            "High weight range indicates dramatic connectivity shifts, "
            "possibly reflecting pressure transients or operational mode changes."
        )
    elif stats['range'] < 0.5:
        lines.append(
            "Stable weight range suggests consistent network topology across the observation window."
        )
    
    return ' '.join(lines)


def assess_risk_level_and_confidence(
    pattern: str,
    stats: dict
) -> Tuple[str, float]:
    """
    Map network pattern to domain risk level and confidence.
    
    High confidence = consistent, interpretable patterns.
    Low confidence = uncertain or transitional states.
    """
    base_confidence = {
        'high_stability': 0.92,
        'moderate_connectivity': 0.85,
        'sparse_network': 0.78,
        'high_volatility': 0.75,
        'mixed_dynamics': 0.82,
        'unknown_pattern': 0.60,
    }.get(pattern, 0.60)
    
    # Adjust confidence based on variability
    if stats['std'] > 0.6:
        base_confidence *= 0.85  # Reduce confidence if very noisy
    
    # Assign risk level based on connectivity mean
    mean_w = stats['mean']
    if mean_w > 0.5:
        risk = 'high_risk'  # Strong connectivity = high reliance on network
    elif mean_w > 0.25:
        risk = 'moderate_risk'
    else:
        risk = 'low_risk'  # Weak connectivity = low systemic risk
    
    return risk, base_confidence


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("GNN Insights Extraction Pipeline")
    print("="*70)
    
    # Load data
    print(f"\n1. Loading edge weights from {EDGE_WEIGHTS_PATH}...")
    if not EDGE_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Edge weights file not found: {EDGE_WEIGHTS_PATH}")
    
    edge_weights_all = np.load(EDGE_WEIGHTS_PATH)
    print(f"   Shape: {edge_weights_all.shape}")
    print(f"   (epochs, windows, window_steps, edges, 1)")
    
    print(f"\n2. Loading sensor data from {SENSOR_DATA_PATH}...")
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    print(f"   Rows: {len(sensor_df)}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract records
    print(f"\n3. Extracting insights (max {MAX_RECORDS} records)...")
    records = []
    
    # Flatten edges: use last epoch, aggregate across windows
    last_epoch_weights = edge_weights_all[-1]  # [num_windows, window_steps, num_edges, 1]
    num_windows = last_epoch_weights.shape[0]
    window_steps = last_epoch_weights.shape[1]
    
    # Create one summary per window (aggregate across time steps within window)
    for window_idx in range(min(num_windows, MAX_RECORDS)):
        window_data = last_epoch_weights[window_idx]  # [window_steps, num_edges, 1]
        
        # Aggregate edge weights across time steps in this window
        aggregated_weights = np.mean(window_data, axis=0)  # [num_edges, 1]
        
        # Extract statistics
        stats = extract_window_statistics(aggregated_weights)
        pattern, _ = classify_connectivity_pattern(stats)
        risk_level, confidence = assess_risk_level_and_confidence(pattern, stats)
        
        # Find top edges
        weights_flat = aggregated_weights.flatten()
        top_edge_indices = np.argsort(weights_flat)[::-1][:5]
        top_edges = [
            (i, i+1, normalize_weight(weights_flat[i]))  # Simple pairs for demo
            for i in top_edge_indices[:3]
        ]
        
        # Optional sensor snapshot (align with window index if possible)
        sensor_snapshot = None
        if window_idx < len(sensor_df):
            row = sensor_df.iloc[window_idx]
            sensor_snapshot = {
                'net_flow': row.get('net_flow', 0),
                'injection_pressure': row.get('injection_pressure', 0),
            }
        
        # Generate narrative
        text = generate_narrative_text(
            sample_idx=window_idx,
            stats=stats,
            pattern=pattern,
            top_edges=top_edges,
            sensor_snapshot=sensor_snapshot
        )
        
        records.append({
            'sample_id': window_idx + 1,
            'text': text,
            'label': risk_level,
            'confidence': confidence,
            'pattern': pattern,
            'edge_weight_mean': stats['mean'],
            'edge_weight_std': stats['std'],
            'active_edges': stats['active_edges'],
        })
    
    df_output = pd.DataFrame(records)
    
    # Save
    df_output.to_csv(OUTPUT_CSV, index=False)
    print(f"\n4. Saved {len(df_output)} records to {OUTPUT_CSV}")
    
    # Summary statistics
    print("\n5. Summary Statistics:")
    print(f"   Total records: {len(df_output)}")
    print(f"\n   Label distribution:")
    print(df_output['label'].value_counts().to_string())
    print(f"\n   Confidence stats:")
    print(f"   Mean: {df_output['confidence'].mean():.3f}")
    print(f"   Min:  {df_output['confidence'].min():.3f}")
    print(f"   Max:  {df_output['confidence'].max():.3f}")
    
    print(f"\n   Sample records:")
    print(df_output[['sample_id', 'label', 'confidence', 'pattern']].head(10).to_string())
    
    print("\n" + "="*70)
    print("Extraction complete. Ready for Week 4 LLM notebooks.")
    print("="*70)
    
    return df_output


if __name__ == '__main__':
    df = main()
