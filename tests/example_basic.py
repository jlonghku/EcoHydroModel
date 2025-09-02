import torch
from ecohydromodel import EcoHydroModel, DataManager, Visualizer, Trainer
from custom_updater import BucketUpdater

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Data ---
data = DataManager(device)
dem = 'data/dem.asc'
data.load_dem(dem, outx=38, outy=54)

ref = {'storage': 'data/ref_storage_bucket.csv'}
data.load_csv(ref, target='ref', time_index=[1, 365])

forcing = {'RAIN': 'data/ref_RAIN.csv', 'TEMP': 'data/ref_TEMP.csv'}
data.load_csv(forcing, target='forcing', time_index=[1, 365])

state = {'storage': 'data/ref_storage_bucket.csv'}
data.load_csv(state, target='state', time_index=0)

# --- Model ---
bucket = BucketUpdater().to(device)
model = EcoHydroModel({'bucket': bucket}, device=device)
var = 'storage'

# --- Test run ---
model.eval()
with torch.no_grad():
    out = model(data)

Visualizer.plot(out[var], pos=data.pos)
Visualizer.plot_similarity(out[var], data.ref[var], pos=data.pos)
Visualizer.plot_timeseries(out[var], data.ref[var], node_idx=0)

# --- Training ---
print("Start training...")
final_params = Trainer(model).train(data, epochs=100, lr=1e-2, chunk_size=1)
print('Final params:', final_params)

# --- Evaluation ---
model.eval()
with torch.no_grad():
    out = model(data)

Visualizer.plot(out[var], pos=data.pos)
Visualizer.plot_similarity(out[var], data.ref[var], pos=data.pos)
Visualizer.plot_timeseries(out[var], data.ref[var], node_idx=0)