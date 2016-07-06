import numpy as np
import sympy as sp

def init(u, x, gamma):
    """
    Initialize variables
    """
    for i, x_i in enumerate(x):
        if (x_i < 0):
            u[i, 0] = 1
            u[i, 1] = 0
            u[i, 2] = 100000/(gamma-1) 
        else:
            u[i, 0] = 0.125
            u[i, 1] = 0
            u[i, 2] = 10000/(gamma-1)
    return u

def getFlux(f, u, gamma):
    """
    Get flux from variables
    """
    assert(f.shape[0]==u.shape[0])
    f[:,0] = u[:,1]
    f[:,1] = u[:,1]*u[:,1]/u[:,0] + (gamma-1)*(u[:,2]-0.5*u[:,1]*u[:,1]/u[:,0])
    f[:,2] = (u[:,1]/u[:,0])*(u[:,2] + (gamma-1)*(u[:,2]-0.5*u[:,1]*u[:,1]/u[:,0]))
#    f[0] = u[1]
#    f[1] = u[1]*u[1]/u[0] + (gamma-1)*(u[2]-0.5*u[1]*u[1]/u[0])
#    f[2] = (u[1]/u[0])*(u[2] + (gamma-1)*(u[2]-0.5*u[1]*u[1]/u[0]))

    return f


def getRichtMyerAdvance(u, dt, dx, gamma):
    """
    Advance solution based on the RichtMyer method
    """
    uHalf = np.zeros_like(u)
    f = np.zeros_like(u)
    fHalf = np.zeros_like(f)

    f = getFlux(f, u, gamma)
    uHalf[:-1] = 0.5*(u[1:]+u[:-1]) - 0.5*(dt/dx)*(f[1:]-f[:-1])
    fHalf = getFlux(fHalf, uHalf, gamma)

    uHalf[1:-1] = u[1:-1] - (dt/dx)*(fHalf[1:-1]-fHalf[:-2])
    return uHalf


def getBc(u, gamma):
    u[0, 0] = 1
    u[0, 1] = 0
    u[0, 2] = 100000/(gamma-1)
    u[-1, 0] = 0.125
    u[-1, 1] = 0
    u[-1, 2] = 10000/(gamma-1)

    return u


xMin = -10
xMax =  10

nx = 1001
dx = (xMax-xMin)/(nx-1)

x = np.linspace(xMin,xMax, nx)
u = np.zeros((nx, 3), dtype = float)
f = np.zeros((nx, 3), dtype = float)

gamma = 1.4
u = init(u, x, gamma)
sigma = 0.45

timesteps = 1000
for i in range(timesteps):
    p = (gamma-1)*(u[:,2]-0.5*u[:,1]**2/u[:,0])
    cMax = np.sqrt(gamma*max(p/u[:,0]))
    uMax = max((u[:,1]/u[:,0]))
    dt = sigma*dx/(cMax+uMax)
    u = getRichtMyerAdvance(u, dt, dx, gamma)
    u = getBc(u, gamma)

import matplotlib.pyplot as plt

#fig, ax = plt.subplots(1,3)
fig = plt.subplot()
#fig.set_ylim(-0.2, 1.2)
plt.plot(x,u[:,0])
#plt.plot(x,u[:,1]/u[:,0])
#plt.plot(x, np.sqrt(gamma*(p/u[:,0])))
#ax[0].plot(x,u[:,0], ls = "--")
#ax[1].plot(x,u[:,1]/u[:,0], ls = "-.")
#ax[2].plot(x,(gamma-1)*(u[:,2]-0.5*u[:,1]*u[:,1]/u[:,0]), ls = "-")
plt.show()
