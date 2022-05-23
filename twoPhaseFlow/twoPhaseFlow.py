from scipy.optimize import brentq as root
import cmath
from pynverse import inversefunc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc

tol = 1e-12

def u_solve(u_nonconvex, x, t, M):
    u_sol = np.array([np.array([u_nonconvex(xi, ti, M) for xi in x]) for ti in t])
    return u_sol

def diff(f, u, M, h=tol):
  # complex-step differentiation
  c = complex(u, h)
  df = f(c,M).imag/h
  return df

def f_w(u, M):
  return u**2/(u**2+(1-u)**2/M)

def df_w(u, M):
  #return diff(f_w, u, M)
  return -2*M*(u-1)*u/(((M+1)*u**2-2*u+1)**2)
  #return u**2

def u_nonconvex(x, t, M):
  f = lambda u,M: (f_w(u, M) - df_w(0, M))/(u - 0) - df_w(u, M)
  ustar = root(f, 0+tol,1-tol, args=(M))
  df_wustar = df_w(ustar, M)
  df_w1 = df_w(1, M)
  df_w_inv = inversefunc(df_w, domain=[ustar,1], args=(M))

  if t == 0:
      return 0
  if x/t > df_wustar:
    return 0
  elif df_wustar >= x/t and x/t >= df_w1:
    return float(df_w_inv(x/t))
  elif df_w1 >= x/t:
    return 1

def u_concave(x, t, M):
  if t == 0:
      return 0
  if x/t > M:
    return 0
  elif M >= x/t and x/t >= 1/M:
    return (np.sqrt(M*t/x)-1)/(M-1)
  elif 1/M >= x/t:
    return 1

def u_convex(x, t, M):
  if t == 0:
    return 0
  if x/t <= 1:
    return 1
  elif x/t > 1:
    return 0


if __name__ == '__main__':
  M = 1
  x = 0
  tplot = [0.25, 0.5, 0.75]
  Nt = 100
  t = np.linspace(0, 1, Nt+1)
  N = 100
  x = np.linspace(0,1,N)

  u_sol = np.array([np.array([u_nonconvex(xi, ti, M) for xi in x]) for ti in t])

  fig, ax = plt.subplots(1,3, figsize=(10,3), sharey=True)

  for axi,tploti in zip(ax,tplot):
    idx = np.where(t==tploti)[0][0]
    axi.plot(x,u_sol[idx])
    axi.set_xlabel(r'x')
    axi.set_ylabel(r'u(x,t)')
    axi.set_title(r't={:.2f}'.format(tploti))

  plt.show()


  fig = plt.figure(figsize=(5,5))
  ax = plt.axes(xlim=(0.0,1.0),ylim=(0.0-0.05,1.0+0.05),xlabel=(r'x'),ylabel=(r'u(x,t)'))
  line = ax.plot([], [], lw=1)[0]

  def init():
      line.set_data([], [])
      return line,

  def animate(i):
      y = u_sol[i].copy()
      line.set_data(x,y)   
      ax.set_title(r't={:.2f}'.format(t[i]))
      return line,

  anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t.size, interval=100, blit=True)
  plt.close()
  anim

  M = 2
  tplot = [0.25, 0.5, 0.75]
  N = 100
  x = np.linspace(0,1,N)
  Nt = 100
  t = np.linspace(0, 1, Nt+1)

  u_sol = np.array([np.array([u_concave(xi, ti, M) for xi in x]) for ti in t])

  fig, ax = plt.subplots(1,3, figsize=(10,3), sharey=True)

  for axi,tploti in zip(ax,tplot):
    idx = np.where(t==tploti)[0][0]
    axi.plot(x,u_sol[idx])
    axi.set_xlabel(r'x')
    axi.set_ylabel(r'u(x,t)')
    axi.set_title(r't={:.2f}'.format(tploti))

  plt.show()

  fig = plt.figure(figsize=(5,5))
  ax = plt.axes(xlim=(0.0,1.0),ylim=(0.0-0.05,1.0+0.05),xlabel=(r'x'),ylabel=(r'u(x,t)'))
  line = ax.plot([], [])[0]

  def init():
      line.set_data([], [])
      return line,

  def animate(i):
      y = u_sol[i].copy()
      line.set_data(x,y)   
      ax.set_title(r't={:.2f}'.format(t[i]))
      return line,

  anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t.size, interval=100, blit=True)
  plt.close()
  anim

  tplot = [0.25, 0.5, 0.75]
  N = 100
  x = np.linspace(0,1,N)
  Nt = 100
  t = np.linspace(0, 1, Nt+1)

  u_sol = np.array([np.array([u_convex(xi, ti) for xi in x]) for ti in t])

  fig, ax = plt.subplots(1,3, figsize=(10,3), sharey=True)

  for axi,tploti in zip(ax,tplot):
    idx = np.where(t==tploti)[0][0]
    axi.plot(x,u_sol[idx])
    axi.set_xlabel(r'x')
    axi.set_ylabel(r'u(x,t)')
    axi.set_title(r't={:.2f}'.format(tploti))

  plt.show()

  fig = plt.figure(figsize=(5,5))
  ax = plt.axes(xlim=(0.0,1.0),ylim=(0.0-0.05,1.0+0.05),xlabel=(r'x'),ylabel=(r'u(x,t)'))
  line = ax.plot([], [])[0]

  def init():
      line.set_data([], [])
      return line,

  def animate(i):
      y = u_sol[i].copy()
      line.set_data(x,y)   
      ax.set_title(r't={:.2f}'.format(t[i]))
      return line,

  anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t.size, interval=100, blit=True)
  plt.close()
  anim
