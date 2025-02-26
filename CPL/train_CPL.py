import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from functions import nth_derivative

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define training domain
zi, zf = 0.0, 3.0
omega_0i, omega_0f = -2.0, 0.0
omega_ai, omega_af = -2.0, 2.0

# Generate training data
T = torch.cartesian_prod(
    torch.linspace(zi, zf, 80),
    torch.linspace(omega_0i, omega_0f, 10),
    torch.linspace(omega_ai, omega_af, 10)
).to(device)

T = T[torch.randperm(T.shape[0])]  # Shuffle dataset

# Define neural network architecture
def create_ann(n_nodes):
    return nn.Sequential(
        nn.Linear(3, n_nodes), nn.Tanh(),
        nn.Linear(n_nodes, n_nodes), nn.Tanh(),
        nn.Linear(n_nodes, n_nodes), nn.Tanh(),
        nn.Linear(n_nodes, 1)
    ).to(device)

ANN = create_ann(30)

# Reparametrization function
def Param(T, net=ANN, ti=zi):
    """ Applies reparametrization to the neural network """
    out = net(T)
    b = 1 - torch.exp(ti - T[:, 0].view(-1, 1))
    return b * out  # Ensuring initial condition is always satisfied

# Define loss function
Om0, Ol0 = 0.3, 0.7  # Cosmological parameters

def cost(T):
    """ Computes the loss function """
    z, omega_0, omega_a = T[:, 0].view(-1, 1), T[:, 1].view(-1, 1), T[:, 2].view(-1, 1)
    
    a = Om0 * (z + 1) ** 3
    b = Ol0 * (z + 1) ** (3 * (omega_0 + omega_a + 1.0))
    c = torch.exp(-3 * omega_a * z / (z + 1.0))

    DL = nth_derivative(Param, T, 0, 0, 1)  # First derivative
    osc = DL - 1 / torch.sqrt(a + b * c)

    return torch.mean(osc ** 2)

# Training loop
epochs = [10000, 2000]
learning_rates = [0.01, 0.001]
errors = []

for lr, ep in zip(learning_rates, epochs):
    optimizer = torch.optim.Adam(ANN.parameters(), lr=lr)
    pbar = tqdm.tqdm(range(ep), desc="Training", colour='cyan', ncols=100)
    
    for _ in pbar:
        loss = cost(T)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        errors.append(float(loss))
        pbar.set_postfix({'loss': loss.item()})

# Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(range(np.sum(epochs)), errors)
plt.xlabel('Epochs', size=16)
plt.ylabel('$\mathcal{L}_{d_L}$', size=18)
plt.title('CPL Training - 30 Nodes', size=16)
plt.yscale('log')
plt.savefig('Training_CPL_30nodes.pdf', dpi=300)
plt.show()

# Save model
torch.save(ANN.state_dict(), 'CPL_param30_inference')




