"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
#import tensorflow as tf
#from deepxde.backend import tf
import deepxde as dde
import numpy as np
import sys
sys.path.insert(0, '/home/ppiper/MEGA/github/IM458-B/kuramotoSivashinsky')
import KS
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import optuna
from ks_solution import generate_solution as ks_sol


def spaceTimeDomain(x, t):
    o = np.ones(len(x)*len(t))
    t = chunks_multiply(o, len(x), t)
    
    X = np.zeros_like(t)
    X = chunks_copy(X, len(x), x)
    X = np.stack((X, t), axis=1)
    return X

def gen_testdata():
    data = np.load("./dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

def chunks_multiply(l, n, t):
    n = max(1, n)
    for j,i in enumerate(range(0, len(l), n)): l[i:i+n]=l[i:i+n]*t[j] 
    return l

def chunks_copy(l, n, x):
    n = max(1, n)
    for i in range(0, len(l), n): l[i:i+n]=x
    return l

def impulse_like(vec):
    imp = np.zeros_like(vec)
    imp[0]=1.0
    return imp

def solve_exact(f, x, t, M):
    # exact flux
    if f == f_convex:
        f = tpf.u_convex
    elif f == f_concave:
        f = tpf.u_concave
    elif f == f_nonconvex:
        f = tpf.u_nonconvex
        
    u = tpf.u_solve(f, x, t, M)

    u = u.flatten()
    u = u.reshape((len(u),1))

    return u

def animate(X, y_true, y_pred, x, t):
    rc('animation', html='jshtml')
    N = len(x)
    Nt = len(t)
    
    Xc = chunks(X, N)
    y_truec = chunks(y_true, N)
    y_predc = chunks(y_pred, N)

    fig = plt.figure(figsize=(5,5))
    ax = plt.axes(xlim=(0.0,1.0),ylim=(0.0-0.05,1.0+0.05),xlabel=(r'x'),ylabel=(r'u(x,t)'))
    line = ax.plot([], [], lw=1)[0]
    line2 = ax.plot([], [], lw=1)[0]

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        line.set_label('Exact')
        line2.set_label('PINN')
        legend = plt.legend()
        plt.close()
        return line,line2,legend

    def animate(i):
        ax.set_title(r't={:.2f}'.format(t[i]))
        line.set_data(Xc[i][:,0], y_truec[i])   
        line2.set_data(Xc[i][:,0], y_predc[i])
        return line,line2
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt, interval=100, blit=True)
    plt.close()
    anim
    return anim

class PINN(object):
    def __init__(self):
        pass

    def setDomain(self, x_domain, t_domain, N_domain, N_boundary, N_initial):
        geom = dde.geometry.Interval(*x_domain)
        timedomain = dde.geometry.TimeDomain(*t_domain)
        self.geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        self.N_domain=N_domain
        self.N_boundary=N_boundary
        self.N_initial=N_initial

        return self.geomtime
        
    def setICBC(self, icbc):
        self.icbc = icbc

    def setPDE(self, pde):
        self.pde = pde

    def setTopology(self, NN_topology, NN_activation, NN_init):
        self.NN_topology = NN_topology
        self.NN_activation = NN_activation
        self.NN_init = NN_init

    def setModel(self):
        data = dde.data.TimePDE(
            self.geomtime,
            self.pde, 
            self.icbc, 
            num_domain=self.N_domain, 
            num_boundary=self.N_boundary, 
            num_initial=self.N_initial
        )

        net = dde.nn.FNN(self.NN_topology, self.NN_activation, self.NN_init)
        self.model = dde.Model(data, net)

    def train(self, NN_optimizers, NN_lr, NN_epochs):
        self.setModel()

        l = NN_optimizers.__len__()-1
        for i, (opt, lr, epoch) in enumerate(zip(NN_optimizers, NN_lr, NN_epochs)):
            self.model.compile(opt, lr=lr)
            if i == l:
                self.losshistory, self.train_state = self.model.train(epochs=epoch)
            else:
                self.model.train(epochs=epoch)
    
    def plotTrain(self):
        dde.saveplot(self.losshistory, self.train_state, issave=True, isplot=True)

def tpf_PINN(flux, M, eps, animated=False, outputs=False):
    x_domain = (0, 1)
    t_domain = (0, 0.99)
    #eps = 1e-3
    left_bc = 1
    N_domain = 25600*2
    N_boundary = 80
    N_initial = 160
    NN_topology = [2] + [32]*3 + [1]
    NN_activation = "tanh"
    NN_init = "Glorot normal"
    NN_optimizers = ['adam', 'L-BFGS']
    NN_epochs = [15000, None]
    NN_lr = [1e-3, None]

    def tpf_pde(x, u):
        du_t = dde.grad.jacobian(u, x, i=0, j=1)
        df_u = dde.grad.jacobian(flux(u, M), u, i=0, j=0)
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        return du_t + df_u*du_x - eps*du_xx

    #flux = f_convex
    #M = 1
    pde = tpf_pde

    pinn = PINN()

    geomtime = pinn.setDomain(x_domain, t_domain, N_domain, N_boundary, N_initial)

    bc = dde.icbc.DirichletBC(geomtime, 
    lambda x: left_bc, 
    lambda x, on_boundary: on_boundary and np.isclose(x[0], x_domain[0])
    )

    ic = dde.icbc.IC(geomtime, 
    lambda x: np.zeros_like(x[:,0]), 
    lambda _, on_initial: on_initial
    )

    pinn.setICBC([bc, ic])
    pinn.setTopology(NN_topology, NN_activation, NN_init)
    pinn.setPDE(pde)

    pinn.train(NN_optimizers, NN_lr, NN_epochs)

    if outputs:
        pinn.plotTrain()

    return pinn

def f_concave(u, M):
    return u/(u+(1-u)/M)

def f_convex(u, M):
    return u**2

def f_nonconvex(u, M):
    return u**2.0/(u**2.0+(1.0-u)**2.0/M)

def ks_PINN(animated=False, outputs=False):
    #L = 32*np.pi
    #N = 512
    #dt = 0.2
    #tend = 100
    #tsteps = int(tend/dt)
    #x_domain = (0, L-L/N)
    x_domain = (0, L)
    t_domain = (0, tend)
    N_domain = 25600
    N_boundary = 1
    N_initial = 512
    NN_topology = [2] + [50]*3 + [1]
    NN_activation = "sin"
    NN_init = "Glorot normal"
    NN_optimizers = ['adam', 'L-BFGS']
    NN_epochs = [5000, None]
    NN_lr = [1e-3, None]
    ni = 1.0

    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], L)

    def ks_pde(x, u):
        du_t = dde.grad.jacobian(u, x, i=0, j=1)
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        du_xxxx = dde.grad.hessian(du_xx, x, i=0, j=0)
        #return du_t + ni*du_xxxx + du_xx + u*du_x
        return du_t + ni*du_xxxx + du_xx + 0.5*(du_x)**2
    
    pde = ks_pde
    pinn = PINN()

    geomtime = pinn.setDomain(x_domain, t_domain, N_domain, N_boundary, N_initial)

    bc = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)

    ic = dde.icbc.IC(geomtime, 
    #lambda x: np.cos(x[:,0]/L)*(1. + np.sin(x[:,0]/L)),
    lambda x : np.cos(x[:,0]/(L/2/np.pi))*( 1 + np.sin(x[:,0]/(L/2/np.pi)) ),
    lambda _, on_initial: on_initial
    )

    pinn.setICBC([bc, ic])
    pinn.setTopology(NN_topology, NN_activation, NN_init)
    pinn.setPDE(pde)

    pinn.train(NN_optimizers, NN_lr, NN_epochs)

    if outputs:
        pinn.plotTrain()

    return pinn

def plot(uu, tt, x):
    fig = plt.figure()
    ax = fig.gca()
    tt, x = np.meshgrid(tt, x)
    #surf = ax.contourf(tt, x, uu.transpose(), cmap=cm.coolwarm, linewidth=0)
    surf = ax.pcolormesh(tt, x, uu.transpose())
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


# Numeric
L = 32*np.pi
tend = 100
dt = 0.2
N = 512
x = np.linspace(0, L, N)
t = np.arange(0, tend, dt)

u0 = np.cos(x/(L/2/np.pi))*(1+np.sin(x/(L/2/np.pi)))
uu, tt, xx = ks_sol(x, u0, tmax=tend, h=dt)
    
plot(uu,tt,xx)

X = spaceTimeDomain(x, t)

# PINN
pinn_ks = ks_PINN()

u_pred = pinn_ks.model.predict(X)
uu_pred = u_pred.reshape(uu.shape)

plot(uu_pred, t, x)

import pickle
pinn_file = open('pinn_ks.pkl','wb')
pickle.dump(pinn_ks, pinn_file)
