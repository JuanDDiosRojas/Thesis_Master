import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import quad

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
def load_model(filename, n_nodes=20):
    """Loads a trained PyTorch model if available."""
    if os.path.exists(filename):
        model = nn.Sequential(
            nn.Linear(3, n_nodes), nn.Tanh(),
            nn.Linear(n_nodes, n_nodes), nn.Tanh(),
            nn.Linear(n_nodes, n_nodes), nn.Tanh(),
            nn.Linear(n_nodes, 1)
        ).to(device)
        model.load_state_dict(torch.load(filename, map_location=device))
        model.eval()
        return model
    else:
        print(f"Warning: {filename} not found.")
        return None

ANN_CPL = load_model('CPL_param_dict_50')

# Define domain intervals
zi, zf = 0.0, 3.0
omega_0i, omega_0f = -2.0, 0.0
omega_ai, omega_af = -2.0, 2.0

# Define real solution
Omega_m0, Omega_Lambda0 = 0.3, 0.7

def integrand(z, omega_0, omega_a):
    return 1 / np.sqrt(Omega_m0 * (1 + z)**3 + Omega_Lambda0 * (1 + z)**(3 * (1 + omega_0 + omega_a)) * np.exp(-3 * omega_a * z / (1 + z)))

def DL(z, omega_0, omega_a):
    """Computes the theoretical luminosity distance."""
    result = np.zeros_like(z)
    for i in range(len(z)):
        result[i], _ = quad(integrand, 0, z[i], args=(omega_0, omega_a))
    return result

# Define reparametrization function
def Param(T, net=ANN_CPL, ti=zi):
    """Reparametrization of the network output."""
    out = net(T)
    b = 1 - torch.exp(ti - T[:, 0].view(-1, 1))
    return b * out

# Generate test data
z_test = np.linspace(zi, zf, 40)
omega_0_test, omega_a_test = 0.0, -1.0
plt.plot(z_test, DL(z_test, omega_0_test, omega_a_test), label="Real Solution")

z0 = torch.linspace(zi, zf, 60, device=device)
Omega0 = torch.full_like(z0, omega_0_test, device=device)
Omegaa = torch.full_like(z0, omega_a_test, device=device)
X_test = torch.cat((z0.view(-1, 1), Omega0.view(-1, 1), Omegaa.view(-1, 1)), dim=1)

if ANN_CPL:
    plt.plot(z0.cpu().numpy(), Param(X_test).detach().cpu().numpy(), '--r', label='PINN')

plt.legend()
plt.xlabel("Redshift (z)")
plt.ylabel("Luminosity Distance")
plt.title("Comparison of PINN and Analytical Solution")
plt.show()

# Generate error maps
z_mesh = np.linspace(zi, zf, 100)
O0_mesh = [-2.0, -1.5, -1.0, 0.0]
Oa_mesh = np.linspace(omega_ai, omega_af, 100)

z_param = torch.linspace(zi, zf, 100, device=device)
Oa_param = torch.linspace(omega_ai, omega_af, 100, device=device)

error_maps = []
min_error, max_error = float('inf'), float('-inf')

if ANN_CPL:
    for O0i in O0_mesh:
        error_map = np.ones((100, 100))
        for i in range(100):
            for j in range(100):
                input_tensor = torch.tensor([[z_param[i], O0i, Oa_param[j]]], dtype=torch.float32, device=device)
                y_pred = Param(input_tensor, net=ANN_CPL).detach().cpu().numpy()
                y_actual = DL([z_mesh[i]], O0i, Oa_mesh[j])
                
                error = abs(y_pred - y_actual) / abs(y_actual) * 100.0
                error_map[i, j] = error

                min_error = min(min_error, error)
                max_error = max(max_error, error)
        
        error_maps.append(error_map)

# Plot error maps
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Error Map: CPL Model (20 Nodes)', fontsize=18)

for i, o0 in enumerate(O0_mesh):
    ax = axes[i // 2, i % 2]
    mesh_plot = ax.pcolormesh(z_mesh, Oa_mesh, error_maps[i], cmap='inferno', vmin=min_error, vmax=max_error)
    ax.set_title(f"$\omega_0 =$ {o0}")
    if i in [2, 3]: 
        ax.set_xlabel('$z$', size=16)
    if i in [0, 2]: 
        ax.set_ylabel('$\omega_a$', size=16)

fig.colorbar(mesh_plot, ax=axes.ravel().tolist(), format='%1.3f%%')
plt.show()
fig.savefig('CPL_ErrorMaps.pdf', dpi=300)














