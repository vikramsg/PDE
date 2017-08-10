import numpy as np
import scipy 
from polynomial import *
import matplotlib.pyplot as plt

class runDG:

    def plot(self, x, u):
        elements = x.shape[0]
        Np       = x.shape[1]

        size     = elements*Np

        xv       = np.zeros((size))
        uv       = np.zeros((size))

        coun = 0
        for i in range(elements):
            for j in range(Np):
                xv[coun] = x[i, j]
                uv[coun] = u[i, j]
                coun     = coun + 1

        plt.plot(xv, uv)

        plt.show()


    def dgMesh(self, elements, startX, stopX, order):
        Np       = order + 1

        grid     = np.zeros((elements, 2))

        dx       = (stopX - startX)/float(elements)

        for i in range(elements):
            grid[i, 0] = startX + i*dx
            grid[i, 1] = startX + (i + 1)*dx

        intPoints  = np.zeros((elements, Np))

        gaussNodes = np.polynomial.legendre.leggauss(Np)[0] 

        for i in range(elements):
            for j in range(Np):
                intPoints[i, j] = 0.5*(1 - gaussNodes[j])*grid[i, 0] + 0.5*(1 + gaussNodes[j])*grid[i, 1] 

        return(intPoints)


    def project(self, fn, x, u):
        elements = x.shape[0]
        Np       = x.shape[1]

        for i in range(elements):
            for j in range(Np):
                u[i, j] = fn(x[i, j])


    def fn(self, x):
        return 1.0 + 0.2*np.sin(np.pi*(x))

    def getDiscontDeri(self, u):
        elements = u.shape[0]
        Np       = u.shape[1]

        du   = np.zeros((elements, Np))

        gaussNodes = np.polynomial.legendre.leggauss(Np)[0] 
        dPhi       = Poly().lagrangeDeri(gaussNodes)

        for i in range(elements):
            du[i] = np.dot(np.array(dPhi), np.array(u[i]))

        return(du)

    def getDiscontBnd(self, u):
        elements = u.shape[0]
        Np       = u.shape[1]
        order    = Np - 1

        l_R = Poly().lagrange_right(order)
        l_L = Poly().lagrange_left (order)

        u_d = np.zeros((elements, 2))

        for i in range(elements):
            u_d[i, 0] = np.dot(l_L, u[i])
            u_d[i, 1] = np.dot(l_R, u[i])

        return(u_d)



    def dg(self, order):
        Np       = order + 1

        elements =  3
        startX   = -1
        stopX    =  1

        intPoints = self.dgMesh(elements, startX, stopX, order)

        u  = np.zeros((elements, Np))

        self.project(self.fn, intPoints, u) # Set initial condition

        du = self.getDiscontDeri(u)         # Get uncorrected derivative
        
        u_d = self.getDiscontBnd(u)         # Get discontinuous values at boundary

#        self.plot(intPoints, u)

#        print(intPoints)


if __name__=="__main__":
    order = 1

    run   = runDG()

    run.dg(order)

#    pol   = Poly()
#    
#    nodes = pol.gaussNodes(order)
#    dPhi  = pol.lagrangeDeri(nodes)

