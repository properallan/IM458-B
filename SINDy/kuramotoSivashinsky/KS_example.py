from scipy.io import loadmat
import sys

from sklearn.svm import SVC
sys.path.append('/home/ppiper/MEGA/github/IM458-B/SINDy/kuramotoSivashinsky')
import os
os.chdir('/home/ppiper/MEGA/github/IM458-B/SINDy/kuramotoSivashinsky')
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

def SVD(X, fSVD=np.linalg.svd, rank=None):
    if fSVD == np.linalg.svd:
        u, s, vT = fSVD(X, full_matrices=False)
        v = vT.T.conj()

    if rank is not None:
        u, s, v = setRank(u, s, v, rank)

    return u, s, v

def energy(s):
    return s/np.sum(s)*100

def cum_energy(s):
    return np.cumsum(energy(s))

def setRank(u, s, v, rank):
    u = u[:,:rank]
    s = s[:rank]
    v = v[:, :rank]

    return u, s, v

def plotKS(t, x, u):
    # Plot u and u_dot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(t, x, u)
    plt.xlabel('t', fontsize=16)
    plt.ylabel('x', fontsize=16)
    plt.title(r'$u(x, t)$', fontsize=16)
    u_dot = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)

    plt.subplot(1, 2, 2)
    plt.pcolormesh(t, x, u_dot)
    plt.xlabel('t', fontsize=16)
    plt.ylabel('x', fontsize=16)
    ax = plt.gca()
    ax.set_yticklabels([])
    plt.title(r'$\dot{u}(x, t)$', fontsize=16)
    plt.show()
'''
# Load data from .mat file
data = loadmat('./data/kuramoto_sivishinky.mat')
t = np.ravel(data['tt'])
x = np.ravel(data['x'])
u = data['uu']
dt = t[1] - t[0]
dx = x[1] - x[0]

plotKS(t,x,u)


u = u.reshape(len(x), len(t))
u_dot = u_dot.reshape(len(x), len(t))

U, S, VT = np.linalg.svd(u, full_matrices=False)


# Define PDE library that is quadratic in u, and
# fourth-order in spatial derivatives of u.
library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]
pde_lib = ps.PDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    derivative_order=4,
    spatial_grid=x,
    include_bias=True,
    is_uniform=True,
    periodic=True
)

# Again, loop through all the optimizers
print('STLSQ model: ')
optimizer = ps.STLSQ(threshold=10, alpha=1e-5, normalize_columns=True)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u, t=dt)
model.print()

# reconstruct
#U @ np.diag(S) @ VT'''

data = loadmat('data/kuramoto_sivishinky.mat')
t = np.ravel(data['tt'])
x = np.ravel(data['x'])
u = data['uu']

dt = t[1] - t[0]
dx = x[1] - x[0]

plotKS(t,x,u)
rank = 160
U, S, VT = SVD(u)

plt.plot(energy(S)[:rank])
plt.plot(cum_energy(S)[:rank], label='energy {}'.format(cum_energy(S)[:rank][-1]))
plt.legend()

#U, S, V = SVD(u, rank=rank)


X, T = np.meshgrid(x, t)
XT = np.asarray([X, T]).T

# Define PDE library that is quadratic in u, and 
# fourth-order in spatial derivatives of u.
library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]
pde_lib = ps.WeakPDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    derivative_order=4,
    spatiotemporal_grid=XT,
    include_bias=True,
    is_uniform=False,
    periodic=True
)

# Run SR3 with L0 norm
n_models = 10
#optimizer = ps.SR3(threshold=1, max_iter=1, tol=1e-3, 
#                   thresholder='l0', normalize_columns=True)

pde_lib = ps.PolynomialLibrary(degree=6)
optimizer = ps.STLSQ(max_iter=1,threshold=0.001)

model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
u = u.reshape(u.shape + (1,))
model.fit(u)
#model.fit(V)
model.print()
#model.fit(u.reshape(u.shape + (1,)))
#model.fit(u, ensemble=True, 
#          n_models=n_models, n_subset=len(time) // 2, quiet=True)


'''integrator={'atol': 1e-12, 'method': 'LSODA', 'rtol': 1e-12}
V_sindy = model.simulate(V[0],t=t,integrator_kws=integrator)

u_sindy = (U @ np.diag(S) @ V_sindy.T)

plotKS(t,x,u_sindy)'''


