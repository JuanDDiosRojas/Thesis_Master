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

ANN_60 = load_model(create_ann(60), 'LCDM_cost60')
ANN_50 = load_model(create_ann(50), 'LCDM_cost50')
ANN_30 = load_model(create_ann(30), 'LCDM_cost30')
ANN_20 = load_model(create_ann(20), 'LCDM_cost20')

# Filter out any models that were not found
models = [m for m in [ANN_20, ANN_30, ANN_50, ANN_60] if m is not None]
model_names = ['20 nodes', '30 nodes', '50 nodes', '60 nodes'][:len(models)]

# Generate test data
z_test = torch.linspace(zi, zf, 60).to(device)
x0_test = torch.full_like(z_test, 0.2, device=device)
X_test = torch.cat((z_test.view(-1, 1), x0_test.view(-1, 1)), dim=1)

# Analytical solution
y_true = x0_test.cpu().numpy() * (z_test.cpu().numpy() + 1) ** 3

# Plot predictions
plt.figure(figsize=(8, 6))
plt.plot(z_test.cpu().numpy(), y_true, label='Analytical Solution', color='black', linestyle='dashed')

for model, name in zip(models, model_names):
    y_pred = model(X_test).detach().cpu().numpy()
    plt.plot(z_test.cpu().numpy(), y_pred, '--', label=f'PINN {name}')

plt.xlabel("Redshift (z)")
plt.ylabel("Solution")
plt.title("ΛCDM Model Predictions")
plt.legend()
plt.grid(True)
plt.savefig("LCDM_Predictions.pdf", dpi=300)
plt.show()

# Generate error maps
z_mesh = np.linspace(zi, zf, 100)
x0_mesh = np.linspace(x0_i, x0_f, 100)
z_grid = torch.linspace(zi, zf, 100, device=device)
x0_grid = torch.linspace(x0_i, x0_f, 100, device=device)

error_maps = []
min_error, max_error = float('inf'), float('-inf')

for model in models:
    error_map = np.ones((100, 100))
    for i in range(100):
        for j in range(100):
            input_tensor = torch.tensor([[z_grid[i], x0_grid[j]]], dtype=torch.float32, device=device)
            y_pred = model(input_tensor).detach().cpu().numpy()
            y_actual = x0_mesh[j] * (z_mesh[i] + 1) ** 3
            error = abs(y_pred - y_actual) / abs(y_actual) * 100.0
            error_map[i, j] = error

            min_error = min(min_error, error)
            max_error = max(max_error, error)

    error_maps.append(error_map)

# Plot error maps
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Error Map: ΛCDM Model (Cost Function Approach)', fontsize=18)

for i, (error_map, name) in enumerate(zip(error_maps, model_names)):
    ax = axes[i // 2, i % 2]
    mesh_plot = ax.pcolormesh(z_mesh, x0_mesh, error_map, cmap='inferno', vmin=min_error, vmax=max_error)
    ax.set_title(name)
    if i in [2, 3]:
        ax.set_xlabel('$z$', size=16)
    if i in [0, 2]:
        ax.set_ylabel('$\Omega_m$', size=16)

fig.colorbar(mesh_plot, ax=axes.ravel().tolist(), format='%1.1f%%')
plt.show()
fig.savefig('LCDM_ErrorMaps.pdf', dpi=300)





