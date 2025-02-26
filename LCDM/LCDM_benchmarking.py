import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define training domain
zi, zf = 0.0, 3.0
x0_i, x0_f = 0.2, 0.5

# Define architectures
def create_ann(n_nodes):
    return nn.Sequential(
        nn.Linear(2, n_nodes), nn.Tanh(),
        nn.Linear(n_nodes, n_nodes), nn.Tanh(),
        nn.Linear(n_nodes, 1)
    ).to(device)

# Load trained models
def load_model(model, filename):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename, map_location=device))
        model.eval()
        return model
    else:
        print(f"Warning: {filename} not found. Skipping this model.")
        return None

ANN_60 = load_model(create_ann(60), 'L-CDM_param_60')
ANN_50 = load_model(create_ann(50), 'L-CDM_param_dict_50')
ANN_30 = load_model(create_ann(30), 'L-CDM_param_dict_30')
ANN_20 = load_model(create_ann(20), 'L-CDM_param_dict')

# Filter out any models that were not found
models = [m for m in [ANN_20, ANN_30, ANN_50, ANN_60] if m is not None]
model_names = ['20 nodes', '30 nodes', '50 nodes', '60 nodes'][:len(models)]

# Define reparametrization function
def Param(T, net, ti=zi):
    """ Applies the reparametrization technique to meet initial conditions. """
    out = net(T)
    b = 1 - torch.exp(ti - T[:, 0])
    return T[:, 1].view(-1, 1) + b.view(-1, 1) * out

# Generate test grid
z_mesh = np.linspace(zi, zf, 100)
x0_mesh = np.linspace(x0_i, x0_f, 100)
z_grid = torch.linspace(zi, zf, 100, device=device)
x0_grid = torch.linspace(x0_i, x0_f, 100, device=device)

# Generate error maps
error_maps = []
min_error, max_error = float('inf'), float('-inf')

for model in models:
    error_map = np.ones((100, 100))
    for i in range(100):
        for j in range(100):
            input_tensor = torch.tensor([[z_grid[i], x0_grid[j]]], dtype=torch.float32, device=device)
            y_pred = Param(input_tensor, net=model).detach().cpu().numpy()
            y_actual = x0_mesh[j] * (z_mesh[i] + 1) ** 3
            error = abs(y_pred - y_actual) / abs(y_actual) * 100.0
            error_map[i, j] = error

            min_error = min(min_error, error)
            max_error = max(max_error, error)

    error_maps.append(error_map)

# Plot error maps
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Error Map: ΛCDM Model (Reparametrization Approach)', fontsize=18)

for i, (error_map, name) in enumerate(zip(error_maps, model_names)):
    ax = axes[i // 2, i % 2]
    mesh_plot = ax.pcolormesh(z_mesh, x0_mesh, error_map, cmap='inferno', vmin=min_error, vmax=max_error)
    ax.set_title(name)
    if i in [2, 3]:
        ax.set_xlabel('$z$', size=16)
    if i in [0, 2]:
        ax.set_ylabel('$\Omega_m$', size=16)

fig.colorbar(mesh_plot, ax=axes.ravel().tolist(), format='%1.3f%%')
plt.show()
fig.savefig('LCDM_Reparam_ErrorMaps.pdf', dpi=300)
