import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
#import mplcyberpunk
#import tqdm
from functions import nth_derivative#, Param_dirich
# plt.style.use('cyberpunk')

zi=0.0
zf=3.0
#t=torch.linspace(zi,zf,150).view(-1,1)
x0_i=0.2
x0_f=0.5

###################################################################################
nodos=50
ANN_60 = nn.Sequential(nn.Linear(2, 60), nn.Tanh(),
                           nn.Linear(60,60),
                    nn.Tanh(),nn.Linear(60,1))

ANN_50 = nn.Sequential(nn.Linear(2, nodos), nn.Tanh(),
                           nn.Linear(nodos,nodos),
                    nn.Tanh(),nn.Linear(nodos,1))

ANN_30 = nn.Sequential(nn.Linear(2, 30), nn.Tanh(),
                           nn.Linear(30,30),
                    nn.Tanh(),nn.Linear(30,1))

ANN_20 = nn.Sequential(nn.Linear(2, 20), nn.Tanh(),
                           nn.Linear(20,20),
                    nn.Tanh(),nn.Linear(20,1))

ANN_60.load_state_dict(torch.load('L-CDM_param_60'))
ANN_60.eval()

ANN_50.load_state_dict(torch.load('L-CDM_param_dict_50'))
ANN_50.eval()

ANN_30.load_state_dict(torch.load('L-CDM_param_dict_30'))
ANN_30.eval()

ANN_20.load_state_dict(torch.load('L-CDM_param_dict'))
ANN_20.eval()

redes=[ANN_20,ANN_30,ANN_50, ANN_60]
#param
def Param(T,net=ANN_30,ti=zi):
    out = net(T)
    b=1-torch.exp(ti-T[:,0])
    return T[:,1].view(-1,1) +b.view(-1,1)*out

#intervals for the grid
z_mesh = np.linspace(zi, zf, 100)
x0_mesh = np.linspace(x0_i, x0_f, 100)

z_param = torch.linspace(zi, zf, 100)
x0_param = torch.linspace(x0_i, x0_f, 100)
#grid
#mesh=np.ones((100,100))

#The 3 grids to plot for diferents values
mapas=[]
for k in range(4):
    mesh=np.ones((100,100))
    for i in range(100):
        for j in range(100):
            a=Param(torch.tensor([[z_param[i],x0_param[j]]]), net=redes[k]).detach().numpy()
            #a=ANN_param(torch.tensor([[z_param[i],x0_param[j]]])).detach().numpy()
            #b=sol_x([x0_mesh[j],1.0], [z_mesh[i]], 1.5, 3.0)
            b=x0_mesh[j] * (z_mesh[i]+1)**3
            
            mesh[i,j] =  abs(a-b)/abs(b) * 100.0
    mapas.append(mesh)

# Importamos la librería matplotlib
#import matplotlib.pyplot as plt

# Creamos la figura y las subfiguras
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Establecemos el título principal de la figura
fig.suptitle('Error porcentual: $\Lambda-$CDM', fontsize=18)

# Recorremos los mapas y los graficamos
nod=['20 nodos', '30 nodos', '50 nodos', '60 nodos']
for i in range(4):
    ax = axes[i // 2, i % 2]
    
    # Graficamos el mapa de densidad
    pcolormesh = ax.pcolormesh(z_mesh, x0_mesh, mapas[i], cmap='inferno')

    # Añadimos la barra de color individual
    #colorbar = plt.colorbar(pcolormesh, ax=ax)

    # Personalizamos la subfigura
    #ax.set_title(f"$\Omega_0 =$ {o0}")
    ax.set_title(nod[i])
    if i==2 or i==3: ax.set_xlabel('$z$', size=16)
    if i==0 or i==2: ax.set_ylabel('$\Omega_m$', size=16)

fig.colorbar(pcolormesh, ax=axes.ravel().tolist(), format='%1.3f%%')

# # Ajustamos el espacio entre subfiguras
# plt.tight_layout()

# # Mostramos la figura
plt.show()
fig.savefig('LCDM_bench_20_30_50_60_unabarra.pdf')
