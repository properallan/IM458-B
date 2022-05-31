import numpy as np
import sympy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# KSequ.m - solution of Kuramoto-Sivashinsky equation
#
# u_t = -u*u_x - u_xx - u_xxxx, periodic boundary conditions on [0,32*pi]
# computation is based on v = fft(u), so linear term is diagonal
#
# Using this program:
# u is the initial condition
# h is the time step
# N is the number of points calculated along x
# a is the max value in the initial condition
# b is the min value in the initial condition
# x is used when using a periodic boundary condition, to set up in terms of
#   pi
#
# Initial condition and grid setup

def generate_solution(x, u0, tmax=150, h=0.05):
    
    N = x.size
    #x = np.transpose(np.conj(np.arange(1, N+1))) / N
    a = -1
    b = 1
    #u = np.cos(x/16)*(1+np.sin(x/16))
    u = u0
    v = np.fft.fft(u)
    
    # scalars for ETDRK4
    k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2),
                                             np.array([0]), np.arange(-N/2+1, 0))))) / 16
    L = k**2 - k**4
    E = np.exp(h*L)
    E_2 = np.exp(h*L/2)
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    
    # main loop
    uu = np.array([u])
    tt = 0
  
    nmax = round(tmax/h)

    g = -0.5j*k
    
    for n in range(1, nmax):
        
        #print(f'step {n}')
        
        t = n*h
        Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = E_2*v + Q*Nv
        Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E_2*v + Q*Na
        Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E_2*a + Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        
        u = np.real(np.fft.ifft(v))
        uu = np.append(uu, np.array([u]), axis=0)
        tt = np.hstack((tt, t))
        
    return uu, tt, x

def u_anal(xis,temp):

    u_anal = (np.cos(np.pi*xis/20))*(1+np.sin(np.pi*xis/20))
    
    return u_anal    



    return un4
def generate_solution_alternative():
    
    xisi, tempi = sympy.symbols('xisi tempi') #nova função analitica para derivar
    u_anali = (sympy.cos(sympy.pi*xisi/20))*(1+sympy.sin(sympy.pi*xisi/20));
    u_anali
    
    du_analidx = u_anali.diff(xisi)
    du_analidx
    
    from sympy.utilities.lambdify import lambdify
    
    boundary_du = lambdify((xisi, tempi), du_analidx)
    
    #derivarção simbolica da segunda derivada para condiçao inicial
    
    d2u_analidx2 = du_analidx.diff(xisi)
    
    d3u_analidx3 = d2u_analidx2.diff(xisi)
    
    d4u_analidx4 = d3u_analidx3.diff(xisi)
    
    #transformando as derivadas em numeros reais
    
    boundary_d2u = lambdify((xisi, tempi), d2u_analidx2)
    
    boundary_d4u = lambdify((xisi, tempi), d4u_analidx4)
    
    nx = 256
    a=0
    b=200
    L = b-a
    dx = L/(nx-1)
    dt = 0.01
    nt = 175000
    h = (b-a)/(nx-1)
    
    alfa = 1
    gamma = 1
    
    x = np.linspace(a,b,num=nx)
    x = x[:-1] #exclui o ultimo ponto de x
    tempo = np.linspace(0,dt*nt,num=nt)

    t = 0.0
    u0 = np.array([u_anal(xi, t) for xi in x])
    du0 = np.array([boundary_du(xi,t) for xi in x])
    d2u0 = np.array([boundary_d2u(xi,t) for xi in x])
    d4u0 = np.array([boundary_d4u(xi,t) for xi in x])

    A = np.zeros((nx-1,nx-1))

    A[0,-1] = 1
    A[-1,-1] = 3
    A[-1,0] = 1
    A[-1,-1] = 1
    
    for i in range(-1,nx-2):
        i=i+1
        for j in range(0,nx-1):
            if i==j:
                A[i,j] = 3
            if i==j+1:
                A[i,j] = 1
            if i==j-1:
                A[i,j] = 1
            j=j+1
    
    
    A2 = np.linalg.inv(A)
    
    matrixphi = np.zeros((nx-1,nx-1))
    
    for i in range(-1,nx-2):
        i=i+1
        for j in range(0,nx-1):
            if i==j-2:
                matrixphi[i,j] = 1/(12*h)
            if i==j-1:
                matrixphi[i,j] = 28/(12*h)
            if i==j+1:
                matrixphi[i,j] = -28/(12*h)
            if i==j+2:
                matrixphi[i,j] = -1/(12*h)
            j=j+1
            
    matrixphi[-2,-1] = 28/(12*h)
    matrixphi[-2,0] = 1/(12*h)
    matrixphi[-3,-1] = 1/(12*h)
    matrixphi[0,-2] = -1/(12*h)
    matrixphi[0,-1] = -28/(12*h)
    matrixphi[1,-1] = -1/(12*h)
    matrixphi[-1,0] = 28/(12*h)
    matrixphi[-1,1] = 1/(12*h)
    matrixphi[-2,0] = 1/(12*h)
    
    B = np.zeros((nx-1,nx-1))

    B[-1,0] = 1
    B[0,-1] = 1
    
    for i in range(-1,nx-2):
        i=i+1
        for j in range(0,nx-1):
            if i==j:
                B[i,j] = 10
            if i==j+1:
                B[i,j] = 1
            if i==j-1:
                B[i,j] = 1
            j=j+1
    
    B2 = np.linalg.inv(B)
    
    matrixphi2 = np.zeros((nx-1,nx-1))
    
    matrixphi2[-1,0] = 12/(h**2)
    
    for i in range(-1,nx-2):
        i=i+1
        for j in range(0,nx-1):
            if i==j+1:
                matrixphi2[i,j] = 12/(h**2)
            if i==j:
                matrixphi2[i,j] = -(12*2)/(h**2)
            if i==j-1:
                matrixphi2[i,j] = 12/(h**2)
            j=j+1
    
    matrixphi2[0,-1]=12/(h**2)
    
    C = np.zeros((nx-1,nx-1))

    C[-1,0] = 1
    C[0,-1] = 1
    
    for i in range(-1,nx-2):
        i=i+1
        for j in range(0,nx-1):
            if i==j:
                C[i,j] = 10
            if i==j+1:
                C[i,j] = 1
            if i==j-1:
                C[i,j] = 1
            j=j+1
    
    C2 = np.linalg.inv(C)
    
    matrixphi4 = np.zeros((nx-1,nx-1))
    
    matrixphi4[-1,0] = 12/(h**2)
    
    for i in range(-1,nx-2):
        i=i+1
        for j in range(0,nx-1):
            if i==j+1:
                matrixphi4[i,j] = 12/(h**2)
            if i==j:
                matrixphi4[i,j] = -(12*2)/(h**2)
            if i==j-1:
                matrixphi4[i,j] = 12/(h**2)
            j=j+1
    
    matrixphi4[-1,0] = 12/(h**2)
    matrixphi4[0,-1]=12/(h**2)
    
    def avanco_tempo(un,t):
    
        phi2 = np.matmul(matrixphi2,un)
        d22u0 = np.matmul(B2,phi2)
        
        phi = np.matmul(matrixphi,un)
        du0 = np.matmul(A2,phi)
        
        phi4 = np.matmul(matrixphi4,d22u0)
        d44u0 = np.matmul(C2,phi4)
        
        un1 = un + (dt/2)*(-un*du0 - alfa*d22u0 - gamma*d44u0)
    
        phi2 = np.matmul(matrixphi2,un1)
        d22u0 = np.matmul(B2,phi2)
        
        phi = np.matmul(matrixphi,un1)
        du0 = np.matmul(A2,phi)
        
        phi4 = np.matmul(matrixphi4,d22u0)
        d44u0 = np.matmul(C2,phi4)
        
        un2 = un1 + (dt/2)*(-un1*du0 - alfa*d22u0 - gamma*d44u0)
    
        phi2 = np.matmul(matrixphi2,un2)
        d22u0 = np.matmul(B2,phi2)
        
        phi = np.matmul(matrixphi,un2)
        du0 = np.matmul(A2,phi)
        
        phi4 = np.matmul(matrixphi4,d22u0)
        d44u0 = np.matmul(C2,phi4)
        
        un3 = 2*un/3 + un2/3 + (dt/6)*(-un2*du0 - alfa*d22u0 - gamma*d44u0)
    
        phi2 = np.matmul(matrixphi2,un3)
        d22u0 = np.matmul(B2,phi2)
        
        phi = np.matmul(matrixphi,un3)
        du0 = np.matmul(A2,phi)
        
        phi4 = np.matmul(matrixphi4,d22u0)
        d44u0 = np.matmul(C2,phi4)
        
        un4 = un3 + (dt/2)*(-un3*du0 - alfa*d22u0 - gamma*d44u0)
    
        return un4
    
    u_histnumerico =  [u0.copy()]
    u_histanal = [u0.copy()]

    un = u0.copy()

    un1 = un + (dt/2)*(-un*du0 - alfa*d2u0 - gamma*d4u0)

    phi2 = numpy.matmul(matrixphi2,un1)
    d22u0 = numpy.matmul(B2,phi2)

    phi = numpy.matmul(matrixphi,un1)
    du0 = numpy.matmul(A2,phi)

    phi4 = numpy.matmul(matrixphi4,d22u0)
    d4u0 = numpy.matmul(C2,phi4)

    un2 = un1 + (dt/2)*(-un1*du0 - alfa*d22u0 - gamma*d4u0)

    phi2 = numpy.matmul(matrixphi2,un2)
    d22u0 = numpy.matmul(B2,phi2)

    phi = numpy.matmul(matrixphi,un2)
    du0 = numpy.matmul(A2,phi)

    phi4 = numpy.matmul(matrixphi4,d22u0)
    d4u0 = numpy.matmul(C2,phi4)

    un3 = 2*un/3 + un2/3 + (dt/6)*(-un2*du0 - alfa*d22u0 - gamma*d4u0)

    phi2 = numpy.matmul(matrixphi2,un3)
    d22u0 = numpy.matmul(B2,phi2)

    phi = numpy.matmul(matrixphi,un3)
    du0 = numpy.matmul(A2,phi)

    phi4 = numpy.matmul(matrixphi4,d22u0)
    d4u0 = numpy.matmul(C2,phi4)

    un4 = un3 + (dt/2)*(-un3*du0 - alfa*d22u0 - gamma*d4u0)

    u0 = un4.copy()

    for n in range (nt):
        
        un = u0.copy()
        
        u0[:] = avanco_tempo(un, n*dt)
    
        u_analytical = numpy.asarray([u_anal(xi, n*dt) for xi in x])
        
        u_histnumerico.append(un.copy())
        #u_histnumerico[n][:-1] = un
        
        u_histanal.append(u_analytical.copy())
    
def plot():
    
    # plot
    uu, tt, x = generate_solution(N=1024, tmax=150, h=0.05)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    tt, x = np.meshgrid(tt, x)
    surf = ax.plot_surface(tt, x, uu.transpose(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

