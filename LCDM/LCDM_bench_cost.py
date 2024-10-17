import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
#import mplcyberpunk
#import tqdm
#from functions import nth_derivative#, Param_dirich
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

ANN_60.load_state_dict(torch.load('LCDM_cost60'))
ANN_60.eval()

ANN_50.load_state_dict(torch.load('LCDM_cost50'))
ANN_50.eval()

ANN_30.load_state_dict(torch.load('LCDM_cost30'))
ANN_30.eval()

ANN_20.load_state_dict(torch.load('LCDM_cost20'))
ANN_20.eval()

redes=[ANN_20,ANN_30,ANN_50, ANN_60]

pos_ini=0.2

z0 = torch.linspace(zi,zf,60)
x0 = pos_ini*torch.ones([z0.shape[0],1])
#v0 = vel_ini*torch.ones([t0.shape[0],1])
# delta0 = d*torch.ones([t0.shape[0],1])
# omega0 = omega*torch.ones([t0.shape[0],1])

X=torch.cat((z0.view(-1,1),x0),1)
#X.cuda()
#plt.plot(t0, -torch.sin(t0)+2*t0+pos_ini, label='solución real')
#plt.plot(z0.detach(), sol_x([pos_ini,vel_ini], t0, d, omega), label='solución real')
plt.plot(z0,pos_ini * (z0+1)**3, label='Solution')
plt.plot(z0,ANN_20(X).detach().numpy(),'--r', label='PINN')
plt.plot(z0,ANN_30(X).detach().numpy(),'--r', label='PINN')
plt.plot(z0,ANN_50(X).detach().numpy(),'--r', label='PINN')
plt.plot(z0,ANN_60(X).detach().numpy(),'--r', label='PINN')

#plt.plot(t0.view(-1,1),Param(t0.view(-1,1)).detach().numpy(),'--r', label='PINN')
plt.legend()
plt.show()

#param
# def Param(T,net=ANN_param,ti=zi):
#     out = net(T)
#     b=1-torch.exp(ti-T[:,0])
#     return T[:,1].view(-1,1) +b.view(-1,1)*out

#############################################################################################

#intervals for the grid
z_mesh = np.linspace(zi, zf, 100)
x0_mesh = np.linspace(x0_i, x0_f, 100)

z_param = torch.linspace(zi, zf, 100)
x0_param = torch.linspace(x0_i, x0_f, 100)
#grid
# mesh=np.ones((100,100))

# #evaluation of the grid
# for i in range(100):
#     for j in range(100):
#         # a=Param(torch.tensor([[z_param[i],x0_param[j]]]), net=ANN_param).detach().numpy()
#         a=ANN_param(torch.tensor([z_param[i],x0_param[j]])).detach().numpy()
#         #b=sol_x([x0_mesh[j],1.0], [z_mesh[i]], 1.5, 3.0)
#         b=x0_mesh[j] * (z_mesh[i]+1)**3
        
#         mesh[i,j] =  abs(a-b)/abs(b) * 100.0
mapas=[]
for k in range(4):
    mesh=np.ones((100,100))
    for i in range(100):
        for j in range(100):
            #a=Param(torch.tensor([[z_param[i],x0_param[j]]]), net=redes[k]).detach().numpy()
            a=redes[k](torch.tensor([[z_param[i],x0_param[j]]])).detach().numpy()
            #b=sol_x([x0_mesh[j],1.0], [z_mesh[i]], 1.5, 3.0)
            b=x0_mesh[j] * (z_mesh[i]+1)**3
            
            mesh[i,j] =  abs(a-b)/abs(b) * 100.0
    mapas.append(mesh)



# Creamos la figura y las subfiguras
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Establecemos el título principal de la figura
fig.suptitle('Error porcentual: $\Lambda-$CDM costo', fontsize=18)

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

fig.colorbar(pcolormesh, ax=axes.ravel().tolist(), format='%1.1f%%')
# colorbar = plt.colorbar(pcolormesh, format='%1.1f%%')
# colorbar.ax.set_ylabel('error porcentual', size=18)
# ax.set_xlabel('$z$', size=18)
# ax.set_ylabel('$\Omega_{m,0}$', size=18)
# ax.set_title('$\Lambda-CDM$ costo: 20 nodos', size=18)
# Mostramos la gráfica
plt.show()

fig.savefig('LCDM_bench_20_30_50_60_cost_unabarra.pdf')

##########################################################################################

#density map
# fig, ax = plt.subplots()

# # Creamos la barra de densidad
# #pcolormesh = ax.pcolormesh(z_mesh, x0_mesh, mesh, cmap='inferno')

# # Modificamos la función `colorbar` para que muestre el símbolo de porcentaje
# colorbar = plt.colorbar(pcolormesh, format='%1.1f%%')
# colorbar.ax.set_ylabel('error porcentual', size=18)
# ax.set_xlabel('$z$', size=18)
# ax.set_ylabel('$\Omega_{m,0}$', size=18)
# ax.set_title('$\Lambda-CDM$ costo: 20 nodos', size=18)
# # Mostramos la gráfica
# plt.show()
# #plt.savefig()
# # fig.savefig('LCDM_cost20.pdf')




