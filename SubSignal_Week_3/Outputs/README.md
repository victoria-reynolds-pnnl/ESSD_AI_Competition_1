# ST-GNN for Time-Varying Connectivity Inference (10 Pressure Observation Nodes)

This script trains a spatio-temporal graph neural network (ST-GNN) on ~4 months of hourly data to:
1) predict **next-hour pressure** at **10 observation locations** (5 wells × upper/lower intervals), and  
2) learn **edge weights** on a fully connected directed graph (no self-loops) to support **connectivity interpretation over time**.

Inputs come from `ftes_scaled_for_GNN.csv`. Outputs include a saved NumPy array of learned edge weights (`edge_weights_over_time.npy`) and two built-in visualizations (heatmap + circular network plot).

---

## Train/Test Split Strategy

**No train/test split is used, intentionally.**

### Why we do not split (defensible rationale)
The primary objective is **connectivity inference over the observed time period**, not forecasting generalization to unseen future periods. We want learned edge weights available for **all time windows** in the record so we can:
- compare connectivity patterns across different time intervals,
- assess changes around operational regimes (`tc_injecting`, `tc_producing`) and shifts in `delta_P_delta_Q`,
- look for persistent vs transient coupling that may indicate changes in fracture permeability/connectivity.

A conventional chronological train/validation/test split would withhold part of the time series and therefore prevent the model from producing edge weights for that withheld time range, which conflicts with the project goal of describing connectivity **throughout** the monitoring period.

### What we do instead (recommended checks)
Without a split, evaluation is framed as “is the learned representation stable and meaningful?”:
- **Fit-quality checks:** confirm the model learns a non-trivial mapping (loss decreases; predictions are not degenerate).
- **Stability checks:** compare edge-weight patterns across runs/seeds/hyperparameters.
- **Event-based checks:** verify edge-weight changes align with known operational transitions and anomalies.

### When a split would be required
If you want to claim **generalization** (e.g., “predicts future pressures accurately”), then you must add a chronological holdout or rolling-origin evaluation. This script is designed for **inference on the observed record**, not out-of-sample forecasting claims.

---

## Model Performance Summary

### What the model trains on
- **Nodes (10):** pressure observation locations defined by:
  - `tl_interval_pressure`, `tl_bottom_pressure`,
  - `tn_interval_pressure`, `tn_bottom_pressure`,
  - `tc_interval_pressure`, `tc_bottom_pressure`,
  - `tu_interval_pressure`, `tu_bottom_pressure`,
  - `ts_interval_pressure`, `ts_bottom_pressure`

- **Per-node features (4 total in this script):**
  - Base pressure feature (1): the node’s pressure value
  - Engineered features (3):
    - `tc_injecting` (only applied to the node at `tc_interval_pressure`; 0 elsewhere)
    - `tc_producing` (only applied to the node at `tc_interval_pressure`; 0 elsewhere)
    - `delta_P_delta_Q` (broadcast to all nodes)

- **Windowing:**
  - Sliding windows of `window_size = 12` hours are built from the full time series.
  - Model input shape per batch: `[batch, window, num_nodes, node_feat_dim]`.

### Training objective (as implemented)
- The target is the **next-hour pressure** for all nodes, taken from feature index `0` of `node_features_with_engineered`:
  - `target_pressures = node_features_with_engineered[target_idx:target_idx+8, :, 0]`
- Loss: **MSE** between `predicted_pressures` and `target_pressures`.
- Training runs for `num_epochs = 2` in the script.

### What is (and is not) reported
- The script computes training loss but does **not** print/log it.
- There is no baseline comparison, no RMSE/MAE reporting, and no node-wise breakdown.

---

## Failures and Limitations

1. **No generalization claims**
   Without a holdout split, you should not claim out-of-sample forecasting accuracy. This is acceptable for the connectivity-inference objective, but should be stated explicitly (as above).

2. **GRU is computed but not used for the final prediction**
   The model computes `gru_out` but the final pressure prediction uses the last-step node embeddings:
   - `predicted_pressures = pressure_predictor(gcn_out_seq[:, -1])`
   This means the temporal module (GRU) does not directly drive the forecast in the current implementation.

3. **Engineered features are attached to only one node**
   `tc_injecting` and `tc_producing` are applied only to the `tc_interval_pressure` node index; all other nodes receive 0 for these flags. This is a modeling choice that may or may not match the physical influence of operations.

4. **Fully connected directed graph can be dense/noisy**
   All ordered node pairs (excluding self-loops) are used. Without sparsity constraints, learned connectivity may be difficult to interpret and may reflect common-mode trends.

5. **Edge weights are unconstrained**
   Edge weights come from an MLP and can be negative/positive and unbounded, which complicates interpretation as “strength.”

6. **Performance/scalability**
   The custom edge-weighted message passing loops over edges in Python, which can be slow.

7. **Saved edge-weight array shape: use care**
   The script saves edge weights by collecting the **last-time-step edge weights** returned per batch. The resulting saved array is effectively:
   - `edge_weights_over_time.npy`: a NumPy array stacked as `[num_epochs, num_batches, batch_size, num_edges]` *given the current loop structure* (because each batch appends an array of shape `[batch_size, num_edges]`).
   The script comments mention additional dimensions (e.g., `window`) that are not present in what is appended. Treat the saved file as “edge weights collected per epoch and per training batch” and inspect `edge_weights.shape` after loading to confirm.

---

## HITL Review (answers)

1. **Makes sense (face validity):**  
   The inferred connectivity “makes sense” when the consistently highest-weight edges correspond to physically plausible couplings (e.g., upper/lower intervals within the same well, or known interference pairs between wells) and when those edges persist across adjacent time windows rather than appearing as isolated spikes. In practice, we review multiple time indices (and ideally multiple runs) and expect the dominant edges to be stable enough to form a coherent narrative of coupling.

2. **Spurious correlations (confounding risk):**  
   Spurious correlations are a credible risk in this setup because pressures across wells can co-move due to shared operational forcing (injecting/producing state changes) and global field trends. This risk is highest if the “top edges” primarily switch when `tc_injecting`/`tc_producing` toggles or when `delta_P_delta_Q` changes, rather than reflecting stable pathways. We therefore interpret edge weights as interaction scores that require context: edges that only appear during a single operational regime or that mirror common-mode pressure movement are treated as likely confounded rather than true subsurface connectivity.

3. **High-risk failures (what could go wrong):**  
   The highest-risk failure is misinterpreting transient, operation-driven edge-weight changes as structural connectivity changes (e.g., concluding fracture permeability increased/decreased when the effect is actually a short-lived operational response or shared trend). This could mislead operational decisions such as changing injection strategy or prioritizing interventions based on an incorrect coupling map. To reduce this risk, we treat connectivity changes as credible only when they are (a) persistent across multiple consecutive windows, (b) reproducible across runs/seeds, and (c) consistent with known events and independent diagnostics (pressure residual behavior, operational logs, or other surveillance data).

---

## Reproducibility Steps

1. **Install dependencies**
```bash
pip install -r requirements.txt
---
AI was used in several steps in this weeks deliverables including creation of the STGNN and compiling this README.md 
