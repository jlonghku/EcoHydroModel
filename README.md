# EcoHydroModel

EcoHydroModel is a framework built on **PyTorch** and **PyTorch Geometric** for eco-hydrological modeling. It combines **process-based updaters** with **deep learning modules (GNN)** to support watershed DEM extraction, graph construction, state evolution simulation, visualization, and training.  

The framework is particularly suited for **hydrological and biogeochemical processes** (e.g., soil water storage, evapotranspiration, nitrification) and parameter inversion, and can be extended to various **graph-structured process models**.

---

## ✨ Modules
- **Updaters**: Supports hydrological balance, nitrogen cycle, MLP approximation, and can be flexibly extended  
- **DataManager**: Extract watershed from DEM, build directed graph, and manage forcings, states, and references in a unified way  
- **Trainer**: Provides training, checkpointing, parameter constraints, and best parameter tracking  
- **Visualizer**: Supports grid/graph visualization, similarity metrics (NSE/KGE), and time series comparison  

👉 To customize your own updater, see [Updater Specification.md](docs/Updater_Specification.md)

---

## 📦 Installation

```bash
pip install ecohydromodel
```

---

## 🔑 Core Components

### 1. Updater Base Class
- Provides an `update` method to be implemented by subclasses  
- Supports **parallel / layer / converge / max_depth** update modes  
- Includes parameter management, graph aggregation, and iterative convergence utilities  

### 2. DataManager
- `load_dem()`: Extract sub-basin from DEM and build graph structure  
- `load_csv()` / `save_csv()`: Load/save states and reference data  
- `coarse()`: Graph coarsening using graclus  
- `chunk_data()`: Split time series into chunks for training  

### 3. Trainer
- `train()`: Train model with options for target variables, checkpointing, and parameter constraints  
- Automatically saves and restores best parameters  

### 4. Visualizer
- `plot()`: Spatial visualization (grid or graph)  
- `plot_similarity()`: NSE/KGE-based similarity visualization  
- `plot_timeseries()`: Node-level time series comparison  

### 5. Process Update Modules ([custom_updater.py](tests/custom_updater.py))
- **BucketUpdater**: Implements simple Bucket Model with Hamon (1961) evapotranspiration equations  
- **NitriUpdater**: Implements [Del Grosso](docs/Nitrification_Module_Del_Grosso.md) / [Parton](docs/Nitrification_Module_Parton.md) nitrification equations
- **MLPUpdater**: Predicts nitrification rate using soil embedding and MLP  

---

## 🚀 Example Usage ([example_basic.py](tests/example_basic.py))

This example demonstrates how to use **EcoHydroModel** for data loading, model setup, training, and visualization.  
The workflow is divided into five parts: **Environment & Dependencies**, **Data Preparation**, **Model Construction**, **Training**, and **Evaluation & Visualization**.

---

### 1. Data Preparation
```python
data = DataManager(device)
dem = 'data/dem.asc'
data.load_dem(dem, outx=38, outy=54)

ref = {'storage': 'data/ref_storage.csv'}
data.load_csv(ref, target='ref', time_index=[1, 365])

forcing = {'RAIN': 'data/ref_RAIN.csv', 'TEMP': 'data/ref_TEMP.csv'}
data.load_csv(forcing, target='forcing', time_index=[1, 365])

state = {'storage': 'data/ref_storage_bucket.csv'}
data.load_csv(state, target='state', time_index=0)
```
- Initialize the `DataManager` to store and manage input/ref data.  
- Load the DEM and resample to the specified grid size.  
- Load reference data (`ref`), forcing data (`RAIN`, `TEMP`), and initial state.  
- `time_index` can be used to specify a range or a single timestep (if omitted, all timesteps are used).  

---

### 2. Model Construction
```python
bucket = BucketUpdater().to(device)
model = EcoHydroModel({'bucket': bucket}, device=device)
var = 'storage'
```
- Define a simple process module (`BucketUpdater`).  
- Construct the `EcoHydroModel` with the module dictionary.  
- Specify the target variable (`storage`) for training and evaluation.  

---

### 3. Test Run
```python
model.eval()
with torch.no_grad():
    out = model(data)

Visualizer.plot(out[var], pos=data.pos)
Visualizer.plot_similarity(out[var], data.ref[var], pos=data.pos)
Visualizer.plot_timeseries(out[var], data.ref[var], node_idx=0)
```
- Run the model with default setting in evaluation mode to generate predictions.  
- Visualize spatial patterns, similarity with reference data, and node-level time series.  

---

### 4. Training
```python
print("Start training...")
final_params = Trainer(model).train(
    data, epochs=100, lr=1e-2, chunk_size=1, target_keys=[var]
)
print('Final params:', final_params)
```
- Train the model with the given dataset for 100 epochs.  
- Use a simple trainer with learning rate `1e-2` and chunk size of 1.  
- Print the final optimized parameters after training.  

---

### 5. Evaluation & Visualization
```python
model.eval()
with torch.no_grad():
    out = model(data)

Visualizer.plot(out[var], pos=data.pos)
Visualizer.plot_similarity(out[var], data.ref[var], pos=data.pos)
Visualizer.plot_timeseries(out[var], data.ref[var], node_idx=0)
```
- Evaluate the trained model.  
- Generate the same set of visualizations as in the reference run, enabling a direct comparison between model predictions and reference data.  

---

## 🔧 Advanced Usage ([example_full.py](tests/example_full.py))

This example shows **graph coarsening**, **multi-module wiring**, and **constrained parameter tuning**.  
For a full step-by-step description, see the [detailed manual of example](docs/Full_example.md).

### Highlights
- Graph coarsening to switch between resolutions
- Multiple updaters (e.g., `NitriUpdater`, `MLPUpdater`)  
- Parameter range constraints during training  
- Targeted training on selected variables/time windows 

