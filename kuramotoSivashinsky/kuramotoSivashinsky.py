import KS
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# define data and initialize simulation
L    = 200/2/pi
N    = 512
dt   = 0.25
#ninittransients = 10000
#tend = 50000 + ninittransients  
tend = 60000
TL1  = 0.094
dns  = KS.KS(L=L, N=N, dt=dt, tend=tend)

# intial conditions
dns.IC(u0=np.cos(dns.x/dns.L)*(1. + np.sin(dns.x/dns.L)))

# simulate initial transient
dns.simulate()
# convert to physical space
dns.fou2real()

N_plot = 1000
#N_plot = len(dns.tt)
ninit = int(4e4+1e5)
u_plot = dns.uu[ninit:ninit+N_plot,:]
t_plot = (dns.tt[ninit:ninit+N_plot] - dns.tt[ninit])*TL1
#t_plot = (dns.tt[ninit:ninit+N_plot] - dns.tt[ninit])
# Plotting the contour plot
fig = plt.subplots(figsize=(4,8))
t, s = np.meshgrid(t_plot, np.array(range(N))+1)
#t, s = np.meshgrid(t_plot, dns.x/2/pi)
#t, s = np.meshgrid(np.arange(N_plot), np.array(range(N))+1)
#plt.contourf(s, t, np.transpose(u_plot), 15, cmap=plt.get_cmap("seismic"), vmin=-3,vmax=3)
plt.contourf(s, t, np.transpose(u_plot), 15, cmap=plt.get_cmap("seismic"))
plt.colorbar()
plt.ylim([0,10])
#plt.xlim([0,512])

plt.xlabel(r"$N$")
plt.ylabel(r"$t$")
plt.show()  