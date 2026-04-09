import os 
import numpy as np 
import pandas as pd

#os.chdir("C:/Users/fere556/Downloads/SubSignal_Week2/")
df = pd.read_csv('ftes_scaled_for_GNN.csv')

# Prepare Node Feature Tensors
# Suppose your node features are the 10 interval pressures (columns like tl_interval_pressure, ...), plus any engineered features you want to include.
# If you want to include operational flags or ΔP/ΔQ as node features, you can concatenate them as needed.
""" Problem setup
You have 10 nodes (intervals).
For each time step, you want a feature vector for each node.

Step-by-step:

Build your base node features (e.g., interval pressures) as before.
Create arrays for the engineered features:

For tc_injecting and tc_producing:

For the tc node, use the actual values from your DataFrame.
For all other nodes, set to 0.

For delta_P_delta_Q: For every node, use the same value (broadcast). """

num_timesteps = len(df)
num_nodes = 10

# 1. Base node features (e.g., interval pressures)
node_columns = [
    'tl_interval_pressure', 'tl_bottom_pressure',
    'tn_interval_pressure', 'tn_bottom_pressure',
    'tc_interval_pressure', 'tc_bottom_pressure',
    'tu_interval_pressure', 'tu_bottom_pressure',
    'ts_interval_pressure', 'ts_bottom_pressure'
]
node_features = np.stack([df[col].values for col in node_columns], axis=1)  # shape: [num_timesteps, num_nodes]

# 2. Engineered features
engineered = np.zeros((num_timesteps, num_nodes, 3))  # 3 engineered features

# Find the index for the tc node (e.g., 'tc_interval_pressure')
tc_index = node_columns.index('tc_interval_pressure')  # or whichever index is tc

# tc_injecting and tc_producing: only for tc node
engineered[:, tc_index, 0] = df['tc_injecting'].values
engineered[:, tc_index, 1] = df['tc_producing'].values

# delta_P_delta_Q: broadcast to all nodes
engineered[:, :, 2] = df['delta_P_delta_Q'].values[:, np.newaxis]

# 3. Concatenate along the feature dimension
node_features = node_features[..., np.newaxis]  # shape: [num_timesteps, num_nodes, 1]
node_features_with_engineered = np.concatenate([node_features, engineered], axis=-1)
# Final shape: [num_timesteps, num_nodes, 1 + 3]

#  Build the Graph Structure
# For full connectivity, the adjacency matrix is all ones except the diagonal (no self-loops).
# For PyG, you need an edge index:
    
import torch
import itertools
import torch.nn as nn

num_nodes = 10
# All possible pairs except self-loops
edge_index = torch.tensor(
    [list(pair) for pair in itertools.permutations(range(num_nodes), 2)],
    dtype=torch.long
).t()  # shape: [2, num_edges]

class EdgeWeightedGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch, num_nodes, in_dim]
        batch_size, num_nodes, in_dim = x.shape
        out = torch.zeros(batch_size, num_nodes, self.linear.out_features, device=x.device)
        for b in range(batch_size):
            for idx in range(edge_index.size(1)):
                src = edge_index[0, idx]
                tgt = edge_index[1, idx]
                out[b, tgt] += edge_weight[b, idx] * self.linear(x[b, src])
        return out

class STGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, num_nodes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.gcn = nn.Linear(node_feat_dim, hidden_dim)  # Initial node embedding
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.edge_weighted_gcn = EdgeWeightedGCNLayer(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim * num_nodes, hidden_dim, batch_first=True)
        self.pressure_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, edge_index):
        batch_size, window, num_nodes, feat_dim = x_seq.shape
        gcn_out_seq = []
        edge_weights_seq = []
        for t in range(window):
            x_t = x_seq[:, t]  # [batch, num_nodes, feat_dim]
            node_embeds = self.gcn(x_t)  # [batch, num_nodes, hidden_dim]
            # Compute edge weights for all edges
            edge_feats = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_feats.append(torch.cat([node_embeds[:, i, :], node_embeds[:, j, :]], dim=-1))
            edge_feats = torch.stack(edge_feats, dim=1)  # [batch, num_edges, hidden_dim*2]
            edge_weights = self.edge_predictor(edge_feats).squeeze(-1)  # [batch, num_edges]
            edge_weights_seq.append(edge_weights)
            # Edge-weighted message passing
            edge_weights_for_gcn = edge_weights.unsqueeze(-1)  # [batch, num_edges, 1]
            node_embeds = self.edge_weighted_gcn(node_embeds, edge_index, edge_weights_for_gcn)
            gcn_out_seq.append(node_embeds)
        gcn_out_seq = torch.stack(gcn_out_seq, dim=1)  # [batch, window, num_nodes, hidden_dim]
        # Temporal modeling
        gru_in = gcn_out_seq.reshape(batch_size, window, -1)
        gru_out, _ = self.gru(gru_in)
        # Use last time step's node embeddings for pressure prediction
        last_node_embeds = gcn_out_seq[:, -1]  # [batch, num_nodes, hidden_dim]
        predicted_pressures = self.pressure_predictor(last_node_embeds).squeeze(-1)  # [batch, num_nodes]
        # Return edge weights for last time step as well
        last_edge_weights = edge_weights_seq[-1]  # [batch, num_edges]
        return predicted_pressures, last_edge_weights

# Write a function to generate sliding windows of your data    
def create_windows(data, window_size):
    # data: [num_timesteps, num_nodes, node_feat_dim]
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size])
    return np.stack(windows)  # <--- Make sure this line is present!

# Example usage 
import numpy as np

# Assume: model, windows, edge_index are already defined as in previous steps
# windows: [num_windows, window_size, num_nodes, node_feat_dim]
# edge_index: [2, num_edges]

window_size = 12
windows = create_windows(node_features_with_engineered, window_size)
windows = torch.tensor(create_windows(node_features_with_engineered, window_size), dtype=torch.float32)
model = STGNN(node_feat_dim=windows.shape[-1], hidden_dim=32, num_nodes=num_nodes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 2

# To store edge weights for visualization
all_edge_weights = []

import torch.nn.functional as F

for epoch in range(num_epochs):
    model.train()
    epoch_edge_weights = []

    for batch_idx in range(0, len(windows)-8, 8):  # batch size 8
        batch = torch.tensor(windows[batch_idx:batch_idx+8], dtype=torch.float32)
        target_idx = batch_idx + window_size
        if target_idx + 8 > len(node_features_with_engineered):
            break
        target_pressures = torch.tensor(
            node_features_with_engineered[target_idx:target_idx+8, :, 0], dtype=torch.float32
        )  # [batch, num_nodes]
        predicted_pressures, edge_weights = model(batch, edge_index)
        loss = F.mse_loss(predicted_pressures, target_pressures)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Save edge_weights for visualization if desired
        # Save edge_weights for visualization if desired
        edge_weights_np = edge_weights.detach().cpu().numpy()
        epoch_edge_weights.append(edge_weights_np)

    # Concatenate all batches for this epoch
    epoch_edge_weights = np.concatenate(epoch_edge_weights, axis=0)  # [num_windows, window, num_edges, 1]
    all_edge_weights.append(epoch_edge_weights)
    
# After training, save all edge weights as a numpy file
all_edge_weights = np.stack(all_edge_weights, axis=0)  # [num_epochs, num_windows, window, num_edges, 1]
np.save('edge_weights_over_time.npy', all_edge_weights)
print("Saved edge weights to edge_weights_over_time.npy")
    
# Visualizing Edge Weights
# Plot edge weights as a heatmap or line plot to see how connectivity evolves.
# Example using matplotlib:
    
import matplotlib.pyplot as plt
import numpy as np

edge_weights = np.load('edge_weights_over_time.npy')  # shape: [num_epochs, num_windows, window, num_edges, 1]

# Example: visualize for the last epoch
epoch_idx = 1  # or pick the best epoch
weights = edge_weights[epoch_idx,:, :]  # shape: [num_windows, num_edges]

plt.figure(figsize=(12, 6))
plt.imshow(weights.T, aspect='auto', cmap='viridis', vmin = -.5, vmax = 0.5)
plt.xlabel('Time Window')
plt.ylabel('Edge Index')
plt.title('Edge Weights Over Time (Last Epoch)')
plt.colorbar(label='Edge Weight')
plt.show()

#%% Advanced plotting

# Input expectations (adapt to your actual names):

# edge_weights: np.ndarray of shape [E, T, P] (epochs, timesteps, pairs)
# edge_pairs: np.ndarray of shape [P, 2] (node indices for each pair)
# edge_ids: optional list[str] of length P
# N (number of nodes): inferred from edge_pairs if not provided

edge_pairs = np.array(edge_index[:,:]).T
node_names = node_columns

import numpy as np

def infer_num_nodes(edge_pairs: np.ndarray) -> int:
    """Infer number of nodes from max index in pairs."""
    return int(edge_pairs.max()) + 1

def circular_layout(n_nodes: int, radius: float = 1.0, start_angle: float = 0.0):
    """Return positions on a circle: dict {node: (x, y)}."""
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False) + start_angle
    coords = {i: (radius*np.cos(a), radius*np.sin(a)) for i, a in enumerate(angles)}
    return coords

def normalize_weights(w: np.ndarray, eps: float = 1e-12):
    """Min-max normalize to [0,1] per frame."""
    wmin, wmax = np.nanmin(w), np.nanmax(w)
    return (w - wmin) / (wmax - wmin + eps)

def select_edges(weights_frame: np.ndarray, edge_pairs: np.ndarray, edge_ids=None,
                 top_k: int = None, q: float = None):
    """
    Select a subset of edges either by top_k or quantile q (0..1).
    Returns (idx, weights_sel, pairs_sel, ids_sel)
    """
    assert (top_k is None) ^ (q is None), "Specify either top_k or q (quantile), not both."
    order = np.argsort(weights_frame)[::-1]
    if top_k is not None:
        idx = order[:top_k]
    else:
        thr = np.quantile(weights_frame, q)
        idx = np.where(weights_frame >= thr)[0]
    return idx, weights_frame[idx], edge_pairs[idx], (None if edge_ids is None else [edge_ids[i] for i in idx])


import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path

def plot_network_matplotlib(
    edge_weights, edge_pairs, edge_ids=None,
    epoch: int = 0, tstep: int = 0,
    n_nodes: int = None, directed: bool = False,
    top_k: int = 200,  # or set to None and use q=0.95 below
    q: float = None,   # use quantile selection alternative
    radius: float = 1.0, start_angle: float = 0.0,
    node_size: int = 100, node_color: str = "black",
    cmap_name: str = "inferno",
    node_names = None, 
    min_lw: float = 0.1, max_lw: float = 6.0,
    curved: bool = False, curve_strength: float = 0.15,
    figsize=(10,10), show_labels=True, label_fontsize=8, title=None
):
    """
    Plot the epoch/time slice in circular layout with edge thickness ~ weight.
    """
    if n_nodes is None:
        n_nodes = infer_num_nodes(edge_pairs)
    positions = circular_layout(n_nodes, radius=radius, start_angle=start_angle)

    # Extract frame
    W = edge_weights[epoch, tstep, :]  # shape [P]
    Wnorm = normalize_weights(W)
    # Selection
    idx_sel, W_sel, pairs_sel, ids_sel = select_edges(W, edge_pairs, edge_ids, top_k=top_k, q=q)
    Wnorm_sel = Wnorm[idx_sel]

    # Color mapping by raw weight (optional)
    cmap = plt.cm.get_cmap(cmap_name)
    wmin, wmax = float(np.nanmin(W)), float(np.nanmax(W))

    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes
    xs = [positions[i][0] for i in range(n_nodes)]
    ys = [positions[i][1] for i in range(n_nodes)]
    ax.scatter(xs, ys, s=node_size, color=node_color, zorder=3)
    

    if show_labels:
        for i in range(n_nodes):
            label = node_names[i] if node_names is not None else str(i)
            ax.text(
                positions[i][0], positions[i][1],
                label,
                fontsize=label_fontsize,
                ha='center', va='center',
                color='white',
                zorder=4,
                bbox=dict(boxstyle="circle,pad=0.2", fc=node_color, ec="none", alpha=0.8)
            )

    # if show_labels:
    #     for i in range(n_nodes):
    #         ax.text(positions[i][0], positions[i][1], str(i), fontsize=label_fontsize,
    #                 ha='center', va='center', color='white', zorder=4,
    #                 bbox=dict(boxstyle="circle,pad=0.2", fc=node_color, ec="none", alpha=0.8))

    # Draw edges
    for w, wn, (i, j) in zip(W_sel, Wnorm_sel, pairs_sel):
        (x0, y0), (x1, y1) = positions[i], positions[j]
        lw = min_lw + (max_lw - min_lw) * float(wn)
        color = cmap((w - wmin) / (wmax - wmin + 1e-12))

        if not curved:
            if directed:
                # Arrow (straight)
                patch = FancyArrowPatch((x0, y0), (x1, y1),
                                        arrowstyle='-|>', mutation_scale=10+20*wn,
                                        linewidth=lw, color=color, alpha=0.8)
                ax.add_patch(patch)
            else:
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.8, zorder=2)
        else:
            # Quadratic Bezier curve centered off the chord
            mx, my = (x0+x1)/2, (y0+y1)/2
            # Perpendicular offset
            dx, dy = (x1-x0), (y1-y0)
            length = np.hypot(dx, dy) + 1e-12
            nx, ny = -dy/length, dx/length  # normal vector
            cx, cy = mx + curve_strength*length*nx, my + curve_strength*length*ny  # control point

            path_data = [
                (Path.MOVETO, (x0, y0)),
                (Path.CURVE3, (cx, cy)),
                (Path.CURVE3, (x1, y1)),
            ]
            path = Path([p for _, p in path_data], [c for c, _ in path_data])
            if directed:
                # For arrows along path, approximate with end arrow
                patch = FancyArrowPatch(path=path, arrowstyle='-|>',
                                        mutation_scale=10+20*wn, linewidth=lw,
                                        color=color, alpha=0.8)
                ax.add_patch(patch)
            else:
                patch = PathPatch(path, linewidth=lw, edgecolor=color, facecolor='none', alpha=0.8)
                ax.add_patch(patch)

    ax.set_aspect('equal')
    ax.axis('off')
    if title is None:
        title = f"Epoch {epoch}, t={tstep} — Edge thickness encodes weight"
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=wmin, vmax=wmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Edge weight")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Suppose: edge_weights.shape = [E, T, P], edge_pairs.shape = [P, 2]
fig, ax = plot_network_matplotlib(
    edge_weights=edge_weights,
    edge_pairs=edge_pairs,
    node_names = node_names, 
    epoch=1, tstep=500,
    directed=False,       # set True if your edges are directed
    top_k=20,            # or q=0.95
    curved=True,          # nicer readability on dense graphs
    curve_strength=0.2,
    title="GNN connectivity (circular schematic)"
)
