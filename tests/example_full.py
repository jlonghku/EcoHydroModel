import torch
from ecohydromodel import EcoHydroModel, DataManager, Visualizer, Trainer
from custom_updater import NitriUpdater, MLPUpdater

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Data ---
data = DataManager(device)

# Load DEM with extra features
dem = 'data/dem.asc'
extra = {'soil': 'data/soil_types.asc'}
data.load_dem(dem, outx=38, outy=54, extra_asc=extra)

# Reference data (ground truth for calibration)
ref = {
    'humus': 'data/ref_humus.csv',
    'nh4': 'data/ref_nh4.csv',
    'nitrification': 'data/ref_nitrification.csv',
    'storage': 'data/ref_storage.csv',
}
data.load_csv(ref, target='ref', time_index=[1, 365])

# Forcing data (drivers such as rainfall and temperature)
forcing = {
    'RAIN': 'data/ref_RAIN.csv',
    'TEMP': 'data/ref_TEMP.csv',
    'TEMP_SOIL': 'data/ref_TEMP_SOIL.csv',
}
data.load_csv(forcing, target='forcing', time_index=[1, 365])

# Initial state at t0
state = {
    'storage': 'data/ref_storage.csv',
    'humus': 'data/ref_humus.csv',
    'nh4': 'data/ref_nh4.csv',
}
data.load_csv(state, target='state', time_index=0)

# --- Graph Coarsening (optional) ---
# data = data.coarse(3, reduce_dict={'soil': 'mode'})
# data = data.coarse(max_nodes=20, reduce_dict={'soil': 'mode'})

# --- Model ---
mlp = MLPUpdater(hidden_dim=64, num_layers=3).to(device)
nitri = NitriUpdater().to(device)

# Build model with multiple modules
model = EcoHydroModel({'nitri': nitri, 'mlp': mlp}, device=device)

# --- Reference configuration of modules ---
nitri.set_ref_inputs(['storage'])   # Nitri depends on storage
mlp.set_ref_outputs(['nh4', 'humus'])  # MLP predicts nh4 and humus

# --- Test run ---
model.eval()
with torch.no_grad():
    out = model(data)

# Visualization: spatial plots & similarity with reference
var = 'nitrification'
Visualizer.plot(out[var], pos=data.pos)
Visualizer.plot_similarity(out[var], data.ref[var], pos=data.pos)

# --- Training ---
print("Start training...")

# Define parameter ranges for optimization
param_ranges = {
    'nitri.humusNmaxDecay': (0.0001, 0.001),
    'nitri.bulkDensity': (0.5, 2.0),
}

# Train model with target variable
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

# --- Evaluation ---
model.eval()
with torch.no_grad():
    out = model(data)

# Post-training visualization
Visualizer.plot(out[var], pos=data.pos)
Visualizer.plot_similarity(out[var], data.ref[var], pos=data.pos)
