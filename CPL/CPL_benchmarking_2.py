import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
# import mplcyberpunk
# import tqdm
from functions import nth_derivative#, Param_dirich
# plt.style.use('cyberpunk')
from scipy.integrate import quad

#importing model
nodos=20
ANN_CPL =  nn.Sequential(nn.Linear(3, nodos), nn.Tanh(), nn.Linear(nodos,nodos),
                    nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    # nn.Tanh(), nn.Linear(nodos,nodos)
                    nn.Tanh(),nn.Linear(nodos,1))

ANN_CPL.load_state_dict(torch.load('CPL_param_dict_50'))
ANN_CPL.eval()

#Domain intervals
zi=0.0
zf=3.0

omega_0i=-2.0
omega_0f=0.0

omega_ai=-2.0
omega_af=2.0

#########################   real solution   #####################################
# Definimos los parámetros cosmológicos
Omega_m0 = 0.3
Omega_Lambda0 = 0.7
omega_0 = 1.0
omega_a = 1.0

# Función a integrar
def integrand(z):
    return 1 / np.sqrt(Omega_m0 * (1 + z)**3 + Omega_Lambda0 * (1 + z)**(3 * (1 + omega_0 + omega_a)) * np.exp(-3 * omega_a * z / (1 + z)))


def DL(z):
    # Límite inferior de la integral
    z_min = 0.0

    # Límite superior de la integral
    #z_max = 3.0

    # Vector de valores de z
    #z = np.linspace(z_min, z_max, 100)

    # Vector de soluciones de la integral
    resultado = np.zeros_like(z)

    # Calculamos la integral para cada valor de z
    for i in range(len(z)):
        resultado[i], _ = quad(integrand, z_min, z[i])
    return resultado

#Parametrization
def Param(T,net=ANN_CPL,ti=zi):
    out = net(T)
    b=1-torch.exp(ti-T[:,0].view(-1,1))
    return 0.0 + b.view(-1,1)*out

##########################################################################
#solución real
omega_0 = 0.0
omega_a = -1.0
z=np.linspace(0,3,40)
plt.plot(z,DL(z))


z0 = torch.linspace(zi,zf,60)
Omega0 = omega_0*torch.ones([z0.shape[0],1])
Omegaa = omega_a*torch.ones([z0.shape[0],1])
#v0 = vel_ini*torch.ones([t0.shape[0],1])
# delta0 = d*torch.ones([t0.shape[0],1])
# omega0 = omega*torch.ones([t0.shape[0],1])

X=torch.cat((z0.view(-1,1),Omega0,Omegaa),1)
#X.cuda()
#plt.plot(t0, -torch.sin(t0)+2*t0+pos_ini, label='solución real')
#plt.plot(z0.detach(), sol_x([pos_ini,vel_ini], t0, d, omega), label='solución real')
#plt.plot(z0,pos_ini * (z0+1)**3, label='Solution')
plt.plot(z0,Param(X).detach().numpy(),'--r', label='PINN')
plt.show()

############################################################################

#Grid for the percent error
z_mesh = np.linspace(zi, zf, 100)
O0_mesh = [-2.0,-1.5,-1.0,0.0]
Oa_mesh = np.linspace(omega_ai, omega_af, 100)

z_param = torch.linspace(zi, zf, 100)
#O0_param = torch.linspace(x0_i, x0_f, 100)
Oa_param = torch.linspace(omega_ai, omega_af, 100)

mesh=np.ones((100,100))

#The 4 grids to plot for diferents values
# Lista para almacenar los mapas
mapas = []
min_val, max_val = float('inf'), float('-inf')  # Para rastrear el mínimo y el máximo global

# Generamos los mapas
for O0i in O0_mesh:
    mesh = np.ones((100, 100))
    for i in range(100):
        for j in range(100):
            # Evaluamos la red en la cuadrícula
            a = Param(torch.tensor([[z_param[i], O0i, Oa_param[j]]]), net=ANN_CPL).detach().numpy()
            omega_0 = O0i
            omega_a = Oa_mesh[j]
            b = DL([z_mesh[i]])  # Función que define el valor teórico
            
            error = abs(a - b) / abs(b) * 100.0
            mesh[i, j] = error
            
            # Actualizamos el valor mínimo y máximo global
            min_val = min(min_val, error)
            max_val = max(max_val, error)
    
    mapas.append(mesh)

# Creamos la figura y las subfiguras
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Establecemos el título principal de la figura
fig.suptitle('Error porcentual: CPL 20 nodos', fontsize=18)

# Recorremos los mapas y los graficamos
for i, o0 in enumerate(O0_mesh):
    ax = axes[i // 2, i % 2]
    
    # Graficamos el mapa de densidad, con un rango común en la escala de color
    pcolormesh = ax.pcolormesh(z_mesh, Oa_mesh, mapas[i], cmap='inferno', vmin=min_val, vmax=max_val)
    
    # Personalizamos la subfigura
    ax.set_title(f"$\Omega_0 =$ {o0}")
    if i == 2 or i == 3: 
        ax.set_xlabel('$z$', size=16)
    if i == 0 or i == 2: 
        ax.set_ylabel('$\omega_a$', size=16)

# Añadimos una barra de color común para todos los gráficos
fig.colorbar(pcolormesh, ax=axes.ravel().tolist(), format='%1.3f%%')

# Mostramos la gráfica
plt.show()

fig.savefig('CPL20_final.pdf')













