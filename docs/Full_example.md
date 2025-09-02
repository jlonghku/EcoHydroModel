#  Nitri + MLP Example

This guide shows how to: load data → build a model with **NitriUpdater** and **MLPUpdater** → run a reference pass → train with parameter-range constraints → evaluate and visualize.

---

## 1. Data Preparation

```python
import torch
from ecohydromodel import DataManager

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the DataManager to store and manage input/output
data = DataManager(device)

# DEM & ancillary rasters
dem = 'data/dem.asc'
extra = {'soil': 'data/soil_types.asc'}
data.load_dem(dem, outx=38, outy=54, extra_asc=extra)

# Reference (ground-truth) data
ref = {
    'humus': 'data/ref_humus.csv',
    'nh4': 'data/ref_nh4.csv',
    'nitrification': 'data/ref_nitrification.csv',
    'storage': 'data/ref_storage.csv',
}
data.load_csv(ref, target='ref', time_index=[1, 365])

# Forcing (drivers)
forcing = {
    'RAIN': 'data/ref_RAIN.csv',
    'TEMP': 'data/ref_TEMP.csv',
    'TEMP_SOIL': 'data/ref_TEMP_SOIL.csv'
}
data.load_csv(forcing, target='forcing', time_index=[1, 365])

# Initial state
state = {
    'storage': 'data/ref_storage.csv',
    'humus': 'data/ref_humus.csv',
    'nh4': 'data/ref_nh4.csv'
}
data.load_csv(state, target='state', time_index=0)
```

- **DataManager** organizes DEM, ancillary rasters, reference/forcing, and initial state.
- `load_dem(..., outx, outy)` resamples to a target grid.
- `time_index` accepts a range (e.g., `[1, 365]`) or a single step (`0`). If omitted, all timesteps are used.

---

## 2. (Optional) Graph Coarsening

```python
data = data.coarse(1, reduce_dict={'soil': 'mode'})
data = data.coarse(max_nodes=20, reduce_dict={'soil': 'mode'})
```

- Use `coarse(...)` to downscale the graph; `reduce_dict` controls how node/edge attributes are aggregated (e.g., categorical **soil** by `mode`).

---

## 3. Model Construction

```python

mlp = MLPUpdater(hidden_dim=64, num_layers=3).to(device)
nitri = NitriUpdater().to(device)

model = EcoHydroModel({'nitri': nitri, 'mlp': mlp}, device=device)
var = 'nitrification'
```

- Build updaters (here **NitriUpdater** and **MLPUpdater**) and wrap them into **EcoHydroModel**.
- `var` is the target variable used for evaluation/training.

---

## 4. Reference Configuration

```python
nitri.set_ref_inputs(['storage'])
mlp.set_ref_outputs(['nh4', 'humus'])
```

- Declare **NitriUpdater** dependencies on reference **inputs** (e.g., `storage`).
- Declare **MLPUpdater** reference **outputs** it aims to match (e.g., `nh4`, `humus`).

---

## 5. Test Run

```python
model.eval()
with torch.no_grad():
    out = model(data)

Visualizer.plot(out[var], pos=data.pos)
Visualizer.plot_similarity(out[var], data.ref[var], pos=data.pos)
Visualizer.plot_timeseries(out[var], data.ref[var], node_idx=0)
```

- Run a forward pass with default parameters in eval mode.
- Visualize spatial patterns, similarity to the reference, and a node-level time series.

---

## 6. Training

```python
print("Start training...")

param_ranges = {
    'nitri.humusNmaxDecay': (0.0001, 0.001),
    'nitri.bulkDensity': (0.5, 2.0),
}

final_params = Trainer(model).train(
    data,
    epochs=100,
    lr=1e-2,
    chunk_size=1,
    target_keys=[var],
    param_ranges=param_ranges,
    # Optionally, provide another model as reference
    # ref_model=other_model
)

print('nitri.humusNmaxDecay:', final_params['nitri.humusNmaxDecay'])
print('nitri.bulkDensity:', final_params['nitri.bulkDensity'])
```

- Optimize module parameters for 100 epochs with learning rate **1e-2** and `chunk_size=1`.
- `param_ranges` constrains learnable parameters by fully-qualified names (`module.param`).
- Set `ref_model` to use another model’s outputs directly as the training reference data.

---

## 7. Evaluation & Visualization

```python
model.eval()
with torch.no_grad():
    out = model(data)

Visualizer.plot(out[var], pos=data.pos)
Visualizer.plot_similarity(out[var], data.ref[var], pos=data.pos)
Visualizer.plot_timeseries(out[var], data.ref[var], node_idx=0)
```

- Re-run after training and compare with the same figures used in the reference run.

---

## 8. Tips & Notes

- **Reproducibility**: set seeds (`torch.manual_seed(...)`) before training.
- **Device**: all modules are moved to GPU if available (`.to(device)`).
- **Time windows**: narrow `time_index` during prototyping to speed up iteration.
