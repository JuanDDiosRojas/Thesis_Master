import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.integrate import odeint
from functions import nth_derivative  # Import external function

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define training domain
zi, zf = 0.0, 3.0
x0_i, x0_f = 0.2, 0.5
T = torch.cartesian_prod(torch.linspace(zi, zf, 100),
                         torch.linspace(x0_i, x0_f, 40))
T = T[torch.randperm(T.shape[0])].to(device)  # Shuffle training set and move to GPU if available

# Define neural network architecture
n_nodes = 60
ANN = nn.Sequential(nn.Linear(2, n_nodes), nn.Tanh(),
                    nn.Linear(n_nodes, n_nodes), nn.Tanh(),
                    nn.Linear(n_nodes, 1)).to(device)

print(ANN)

def reparametrize(T: torch.Tensor, net: nn.Module, ti: float = zi) -> torch.Tensor:
    """
    Reparametrization function to enforce initial conditions.

    Parameters:
        T (torch.Tensor): Input tensor with shape (n,2) where T[:,0] = z.
        net (torch.nn.Module): Trained neural network.
        ti (float): Reference value for the initial condition.

    Returns:
        torch.Tensor: Reparametrized network output.
    """
    out = net(T)
    b = 1 - torch.exp(ti - T[:, 0])
    return T[:, 1].view(-1, 1) + b.view(-1, 1) * out

def loss_function(t: torch.Tensor, net: nn.Module) -> torch.Tensor:
    """
    Loss function based on the governing differential equation.

    Parameters:
        t (torch.Tensor): Training dataset.
        net (torch.nn.Module): Neural network model.

    Returns:
        torch.Tensor: Loss to minimize.
    """
    x = reparametrize(t, net)
    z = t[:, 0].view(-1, 1)
    Dx = nth_derivative(lambda x: reparametrize(x, net), t, 0, 0, 1)
    osc = Dx - (3 * x / (1.0 + z))
    return torch.mean(osc ** 2)

def train_model(net: nn.Module, T: torch.Tensor, epochs: list, learning_rates: list) -> list:
    """
    Trains the neural network by minimizing the loss function.

    Parameters:
        net (torch.nn.Module): Neural network to train.
        T (torch.Tensor): Training dataset.
        epochs (list): Number of epochs for each learning rate stage.
        learning_rates (list): Learning rate schedule.

    Returns:
        list: Evolution of the loss function during training.
    """
    errors = []
    for k, ep in enumerate(epochs):
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rates[k])
        pbar = tqdm.tqdm(range(ep), desc="Training", colour='cyan', ncols=100)

        for _ in pbar:
            loss = loss_function(T, net)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            errors.append(float(loss))
            pbar.set_postfix({'loss': loss.item()})

    return errors

# Training configuration
epochs = [10000, 40000, 30000]
learning_rates = [0.01, 0.001, 0.0005]
errors = train_model(ANN, T, epochs, learning_rates)

# Training visualization
plt.figure(figsize=(8, 6))
plt.plot(range(np.sum(epochs)), errors)
plt.xlabel('Epochs', size=16, color='pink')
plt.ylabel('$\mathcal{L}_\mathcal{F}$', size=18, color='pink')
plt.title('Training of PINN for $\Lambda$-CDM Model (60 nodes)', size=16, color='pink')
plt.yscale('log')
plt.grid(True)
plt.savefig('training_LCDM_60nodes.pdf', dpi=300)
plt.show()

# Save trained model
torch.save(ANN.state_dict(), 'L-CDM_param_60.pth')
