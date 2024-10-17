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
nodos=30
ANN_CPL =  nn.Sequential(nn.Linear(3, nodos), nn.Tanh(), nn.Linear(nodos,nodos),
                    nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    # nn.Tanh(), nn.Linear(nodos,nodos)
                    nn.Tanh(),nn.Linear(nodos,1))

ANN_CPL.load_state_dict(torch.load('CPL_param30_inferencia'))
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
def integrand(z, omega_0, omega_a):
    return 1 / np.sqrt(Omega_m0 * (1 + z)**3 + Omega_Lambda0 * (1 + z)**(3 * (1 + omega_0 + omega_a)) * np.exp(-3 * omega_a * z / (1 + z)))


def DL(z,omega_0, omega_a):
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
        resultado[i], _ = quad(integrand, z_min, z[i], args=(omega_0,omega_a))
    return resultado

#Parametrization
def Param(T,net=ANN_CPL,ti=zi):
    out = net(T)
    b=1-torch.exp(ti-T[:,0].view(-1,1))
    return 0.0 + b.view(-1,1)*out

##########################################################################
# #plot solución real
# omega_0_0, omega_0_1, omega_0_2, omega_0_3 = -2.0,-1.5,-1.0,0.0
# omega_a_0, omega_a_1, omega_a_2, omega_a_3= -2.0,-1.0,1.0,2.0
# z=np.linspace(0,3,60)
# z0 = torch.linspace(zi,zf,30)
# Omega0 = omega_0_0*torch.ones([z0.shape[0],1])
# Omegaa = omega_a_0*torch.ones([z0.shape[0],1])
# #v0 = vel_ini*torch.ones([t0.shape[0],1])
# # delta0 = d*torch.ones([t0.shape[0],1])
# # omega0 = omega*torch.ones([t0.shape[0],1])
# X1=torch.cat((z0.view(-1,1),omega_0_0*torch.ones([z0.shape[0],1]),omega_a_0*torch.ones([z0.shape[0],1])),1)
# X2=torch.cat((z0.view(-1,1),omega_0_1*torch.ones([z0.shape[0],1]),omega_a_1*torch.ones([z0.shape[0],1])),1)
# X3=torch.cat((z0.view(-1,1),omega_0_2*torch.ones([z0.shape[0],1]),omega_a_2*torch.ones([z0.shape[0],1])),1)
# X4=torch.cat((z0.view(-1,1),omega_0_3*torch.ones([z0.shape[0],1]),omega_a_3*torch.ones([z0.shape[0],1])),1)

# plt.figure(figsize=(8,6))
# #X.cuda()
# #plt.plot(t0, -torch.sin(t0)+2*t0+pos_ini, label='solución real')
# #plt.plot(z0.detach(), sol_x([pos_ini,vel_ini], t0, d, omega), label='solución real')
# #plt.plot(z0,pos_ini * (z0+1)**3, label='Solution')
# plt.plot(z0,Param(X1).detach().numpy(),'o', label='$\omega_0 = -2.0, \omega_a = -2.0 $', color='blue')
# plt.plot(z,DL(z,omega_0_0,omega_a_0), color='darkblue')

# plt.plot(z0,Param(X2).detach().numpy(),'^', label='$\omega_0 = -1.5, \omega_a = -1.0 $', color='teal')
# plt.plot(z,DL(z,omega_0_1,omega_a_1), color='teal')

# plt.plot(z0,Param(X3).detach().numpy(),'*', label='$\omega_0 = -1.0, \omega_a = 1.0 $', color='royalblue')
# plt.plot(z,DL(z,omega_0_2,omega_a_2), color='royalblue')

# plt.plot(z0,Param(X4).detach().numpy(),'+', label='$\omega_0 = 0.0, \omega_a = 2.0 $', color='darkcyan')
# plt.plot(z,DL(z,omega_0_3,omega_a_3), color='darkcyan')

# plt.ylabel('$d_c/H_0$', color='darkblue', size=18)
# plt.xlabel('$z$', color='darkblue', size=18)
# plt.legend()
# # plt.savefig('cpl_solutions.pdf')
# # plt.savefig('cpl_solutions.png')
# plt.show()

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
mapas=[]
for O0i in O0_mesh:
    mesh=np.ones((100,100))
    for i in range(100):
        for j in range(100):
            a=Param(torch.tensor([[z_param[i],O0i, Oa_param[j]]]), net=ANN_CPL).detach().numpy()
            #b=sol_x([x0_mesh[j],1.0], [z_mesh[i]], 1.5, 3.0)
            omega_0 = O0i
            omega_a = Oa_mesh[j]
            #plt.plot(z,DL(z))
            b=DL([z_mesh[i]])
            
            mesh[i,j] =  abs(a-b)/abs(b) * 100.0
    mapas.append(mesh)

#Density map
# Creamos la figura y las subfiguras
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Recorremos los mapas y los graficamos
# Set main title for the entire figure
fig.suptitle('Error porcentual: CPL 30 nodos', fontsize=18)

for i, o0 in enumerate(O0_mesh):
    ax = axes[i // 2, i % 2]
    
    # Graficamos el mapa de densidad
    pcolormesh = ax.pcolormesh(z_mesh, Oa_mesh,mapas[i], cmap='inferno')
    
    # Añadimos la barra de color
    #colorbar = plt.colorbar(pcolormesh, ax=ax)

    # Personalizamos la subfigura
    ax.set_title(f"$\omega_0 =$ {o0}")
    if i==2 or i==3: ax.set_xlabel('$z$', size=16)
    if i==0 or i==2:ax.set_ylabel('$\omega_a$', size=16)

fig.colorbar(pcolormesh, ax=axes.ravel().tolist(), format='%1.3f%%')
#plt.title('percent error: 30 nodes')
#colorbar = plt.colorbar(pcolormesh, format='%1.1f%%')
#fig.colorbar.ax.set_ylabel('percent error', size=13)
# Mostramos la figura
plt.show()
# fig.savefig('CPL30.pdf')













