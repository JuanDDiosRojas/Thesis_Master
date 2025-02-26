import torch
from scipy.integrate import odeint
import numpy as np

def nth_derivative(net: torch.nn.Module, x: torch.Tensor, j: int, i: int, n: int) -> torch.Tensor:
    """
    Computes the nth derivative of the j-th output of a neural network with respect to the i-th input variable.

    Parameters:
        net (torch.nn.Module): Neural network model.
        x (torch.Tensor): Input tensor of shape (n, m).
        j (int): Index of the output component to differentiate.
        i (int): Index of the input variable to differentiate with respect to.
        n (int): Order of the derivative.

    Returns:
        torch.Tensor: Tensor containing the nth derivative of the selected output.
    """
    x.requires_grad = True
    ann = net(x)[:, j].view(-1, 1)
    grad_tensor = torch.ones_like(ann, requires_grad=True)
    
    for _ in range(n):
        ann = torch.autograd.grad(ann, x, grad_outputs=grad_tensor, create_graph=True)[0][:, i].view(-1, 1)
    
    return ann

####################################################################################
# Oscillator equations
def oscilador(Y, t, delta, omega):
    """
    Defines the system of differential equations for a damped harmonic oscillator.

    Parameters:
        Y (list): List containing position x and velocity dx/dt.
        t (float): Time variable.
        delta (float): Damping coefficient.
        omega (float): Angular frequency.

    Returns:
        list: First derivative [dx/dt, d^2x/dt^2].
    """
    x, dx = Y
    return [dx, -delta * dx - omega**2 * x]

def sol_x(y0, t, delta, omega):
    """
    Computes the numerical solution of the oscillator system using scipy's ODE solver.

    Parameters:
        y0 (list): Initial conditions [x0, dx0].
        t (numpy.ndarray): Time array.
        delta (float): Damping coefficient.
        omega (float): Angular frequency.

    Returns:
        numpy.ndarray: Solution for x(t).
    """
    return odeint(oscilador, y0, t, args=(delta, omega))[:, 0]

#####################################################################
def Param(T: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
    """
    Reparametrization function for a neural network to ensure physical constraints.

    Parameters:
        T (torch.Tensor): Input tensor with shape (n, 2) containing independent variables.
        net (torch.nn.Module): Neural network model.

    Returns:
        torch.Tensor: Reparametrized output.
    """
    ti = 0.0
    out = net(T)
    a = (T[:, 0] - ti) * 1.0
    b = 1 - torch.exp(ti - T[:, 0])
    
    return T[:, 1].view(-1, 1) + a.view(-1, 1) + (b.view(-1, 1) ** 2) * out

###################################################################
def Param_dirich(T: torch.Tensor, net: torch.nn.Module, ti: float) -> torch.Tensor:
    """
    Reparametrization to enforce Dirichlet boundary conditions on the neural network output.

    Parameters:
        T (torch.Tensor): Input tensor of shape (n, 2) where T[:, 0] is the independent variable (time).
        net (torch.nn.Module): Neural network model.
        ti (float): Reference time where ANN(ti) = x0.

    Returns:
        torch.Tensor: Reparametrized output satisfying the Dirichlet condition.
    """
    out = net(T)
    b = 1 - torch.exp(ti - T[:, 0])
    
    return T[:, 1].view(-1, 1) + b.view(-1, 1) * out


