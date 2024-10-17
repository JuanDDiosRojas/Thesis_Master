import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
# import mplcyberpunk
import tqdm
from functions import nth_derivative#, Param_dirich
# plt.style.use('cyberpunk')
from scipy.integrate import quad

#Domain intervals
zi=0.0
zf=3.0

omega_0i=-2.0
omega_0f=0.0

omega_ai=-2.0
omega_af=2.0

#t=torch.linspace(zi,zf,50).view(-1,1)

# x0_i=0.2
# x0_f=0.5

T=torch.cartesian_prod(torch.linspace(zi,zf,80),
                       torch.linspace(omega_0i,omega_0f,10),
                       torch.linspace(omega_ai,omega_af,10))

# #random permutation of the training dataset T
T=T[torch.randperm(T.shape[0])]

#Using cuda if available
if torch.cuda.is_available(): T.cuda()

#Network architecture
nodos=30
ANN = nn.Sequential(nn.Linear(3, nodos), nn.Tanh(), nn.Linear(nodos,nodos),
                    nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    # nn.Tanh(), nn.Linear(nodos,nodos)
                    nn.Tanh(),nn.Linear(nodos,1))

def Param(T,net=ANN,ti=zi):
    out = net(T)
    b=1-torch.exp(ti-T[:,0].view(-1,1))
    return 0.0 + b.view(-1,1)*out

#cost function
Om0 = 0.3
Ol0 = 0.7

#omega_0=1
#omega_a=1

def cost(T):
    z=T[:,0].view(-1,1)
    omega_0=T[:,1].view(-1,1)
    omega_a=T[:,2].view(-1,1)


    a = Om0 * (z+1)**3
    b = Ol0*(z+1)**(3*(omega_0 + omega_a +1.0))
    c = torch.exp(-3*omega_a*z/(z+1.0))
    #z=t[:,0].view(-1,1)

    DL = nth_derivative(Param,T,0,0,1)
    osc = DL - 1/torch.sqrt(a + b*c)

    return torch.mean(osc**2)

#Training loop
epochs=[10000,2000]#,3000]
tasas=[0.01,0.001]#,0.0005]
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
plt.xlabel('Épocas', size=16, color='pink')
plt.ylabel('$\mathcal{L}_{d_L}}$', size=18, color='pink')
plt.title('Entrenamiento CPL 30 nodos', size=16, color='pink')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('Entrenamiento CPL 30 nodos.pdf', dpi=300)
#plt.show()
# Guardar
#save model
torch.save(ANN.state_dict(),'CPL_param30_inferencia')




