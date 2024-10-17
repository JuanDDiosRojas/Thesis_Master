import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import mplcyberpunk
import tqdm
from functions import nth_derivative#, Param_dirich
plt.style.use('cyberpunk')


#Domain intervals

zi=0.0
zf=3.0
#t=torch.linspace(zi,zf,150).view(-1,1)
x0_i=0.2
x0_f=0.5

T=torch.cartesian_prod(torch.linspace(zi,zf,100),
                       torch.linspace(x0_i,x0_f,40))

#random permutation of the training dataset T
T=T[torch.randperm(T.shape[0])]

#if avaliable, use cuda
if torch.cuda.is_available(): T.cuda()

#Neural network architecture

nodos=60
ANN = nn.Sequential(nn.Linear(2, nodos), nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    # nn.Tanh(), nn.Linear(nodos,nodos)
                    nn.Tanh(),nn.Linear(nodos,1))
print(ANN)

#Re-parametrization
def Param(T,net=ANN,ti=zi):
    out = net(T)
    b=1-torch.exp(ti-T[:,0])
    return T[:,1].view(-1,1) +b.view(-1,1)*out

#cost function
def cost(t):
    x=Param(t)
    z=t[:,0].view(-1,1)
    Dx = nth_derivative(Param,t,0,0,1)
    osc = Dx - (3*x / (1.0+z))
    return torch.mean(osc**2)

#training loop

#pbar = tqdm.tqdm(range(epocas), desc="Training",  colour='cyan', ncols=100)
epochs=[10000,40000,30000]
tasas=[0.01,0.001,0.0005]
errores=[]
for k in range(len(epochs)):
    learning_rate=tasas[k]
    epocas=epochs[k]

    #optimizer=torch.optim.SGD(ANN.parameters(),lr=learning_rate,momentum=0.9)
    optimizer = torch.optim.Adam(ANN.parameters(), lr=learning_rate)
    pbar = tqdm.tqdm(range(epocas), desc="Training",  colour='cyan', ncols=100)
    for i in pbar:
        l=cost(T) #coste
        l.backward() #gradiente
        optimizer.step() #se actualizan los parámetros
        optimizer.zero_grad() #vacíamos el gradiente
        errores.append(float(l))
        pbar.set_postfix({'loss': l.item()})

plt.figure(figsize=(8, 6))
plt.plot(range(np.sum(epochs)),errores)
plt.xlabel('Epochs', size=16, color='pink')
plt.ylabel('$\mathcal{L}_\mathcal{F}$', size=18, color='pink')
plt.title('Entrenamiento 60 nodos', size=16, color='pink')

# Leyenda
plt.legend(loc='best')
plt.yscale('log')
plt.show()
# Guardar
plt.savefig('training_LCDM_60nodes.pdf', dpi=300)
###########################################################
#plot model
# pos_ini=0.4
# z0 = torch.linspace(zi,zf,60)
# x0 = pos_ini*torch.ones([z0.shape[0],1])
# X=torch.cat((z0.view(-1,1),x0),1)
# plt.plot(z0,pos_ini * (z0+1)**3, label='Solution')
# plt.plot(z0,Param(X).detach().numpy(),'--r', label='PINN')
# plt.legend()
# plt.show()

torch.save(ANN.state_dict(),'L-CDM_param_60')
