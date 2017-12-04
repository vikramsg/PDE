import numpy as np
import scipy 
from polynomial import *
import matplotlib.pyplot as plt

class WaveDG:

    def __init__(self, order, elements, startX, stopX, correcType = 0, correcFac = 0.9):
        self.order    = order
        self.Np       = order + 1

        self.elements = elements 
        self.startX   = startX 
        self.stopX    = stopX

        self.x_r, self.intPoints = self.dgMesh(self.elements, self.startX, self.stopX, self.order)

        self.u   = np.zeros((self.elements, self.Np))
        self.rhs = np.zeros((self.elements, self.Np))
        
        self.project(self.fn, self.intPoints, self.u) # Set initial condition

        self.nodes = np.polynomial.legendre.leggauss(self.Np)[0] 

        if correcType == 0:
            self.dg_l  = Poly().leftRadauDeri(order, self.nodes)
            self.dg_r  = Poly().rightRadauDeri(order, self.nodes)
        elif correcType == 1:
            self.dg_l  = Poly().getKrivodLeft(order, self.nodes, correcFac)
            self.dg_r  = Poly().getKrivodRight(order, self.nodes, correcFac)
        elif correcType == 2:
            self.dg_l  = Poly().getFilterLeft(order, self.nodes)
            self.dg_r  = Poly().getFilterRight(order, self.nodes)
        else:
            raise Exception('CorrecType not supported')

        gaussNodes = np.polynomial.legendre.leggauss(self.Np)[0] 
        self.dPhi  = Poly().lagrangeDeri(gaussNodes)

        self.l_R   = Poly().lagrange_right(order)
        self.l_L   = Poly().lagrange_left (order)

        self.krivod_l_R = Poly().krivodInterpolate(self.nodes, order, correcFac,  1.0) 
        self.krivod_l_L = Poly().krivodInterpolate(self.nodes, order, correcFac, -1.0) 

        van   = Poly().vandermonde(order, self.nodes) 
        vanF  = Poly().filterVandermonde(order, self.nodes)

        self.filterVan  = np.dot(vanF, np.linalg.inv(van)) 



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
        
        x_r    = np.zeros((elements))
        for i in range(elements):
            x_r[i] = (grid[i, 1] - grid[i, 0])/2.0

        intPoints  = np.zeros((elements, Np))

        gaussNodes = np.polynomial.legendre.leggauss(Np)[0] 

        for i in range(elements):
            for j in range(Np):
                intPoints[i, j] = 0.5*(1 - gaussNodes[j])*grid[i, 0] + 0.5*(1 + gaussNodes[j])*grid[i, 1] 

        return(x_r, intPoints)


    def project(self, fn, x, u):
        elements = x.shape[0]
        Np       = x.shape[1]

        for i in range(elements):
            for j in range(Np):
                u[i, j] = fn(x[i, j])


    def getDiscontDeri(self, u):
        elements = u.shape[0]
        Np       = u.shape[1]

        du       = np.zeros_like(u)

        dPhi     = self.dPhi 

        for i in range(elements):
            du[i] = np.dot(np.array(dPhi), np.array(u[i]))

        return(du)

    def getDiscontBnd(self, u):
        elements = u.shape[0]
        Np       = u.shape[1]
        order    = Np - 1

        l_R      = self.l_R 
        l_L      = self.l_L 

        u_d = np.zeros((elements, 2))

        for i in range(elements):
            u_d[i, 0] = np.dot(l_L, u[i])
            u_d[i, 1] = np.dot(l_R, u[i])

        return(u_d)


    def getInteractionValues(self, u_d):
        elements = u_d.shape[0]
        Np       = u_d.shape[1]

        u_I = np.zeros((elements, 2))

        alpha = 0.5

        #Central flux
#        for i in range(1, elements - 1):
#            u_I[i, 0] = alpha*(u_d[i - 1, 1] + u_d[i    , 0] )
#            u_I[i, 1] = alpha*(u_d[i    , 1] + u_d[i + 1, 0] )
#
#        # Periodic boundary
#        u_I[0       , 0] = alpha*(u_d[elements - 1, 1] + u_d[0    , 0] )
#        u_I[0       , 1] = alpha*(u_d[0           , 1] + u_d[0 + 1, 0] )
#
#        u_I[elements - 1, 0] = alpha*(u_d[elements - 2, 1] + u_d[elements - 1, 0] )
#        u_I[elements - 1, 1] = alpha*(u_d[elements - 1, 1] + u_d[0           , 0] )
#
        for i in range(1, elements - 1):
            u_I[i, 0] = u_d[i - 1, 1] 
            u_I[i, 1] = u_d[i    , 1] 

        # Periodic boundary
        u_I[0       , 0] = u_d[elements - 1, 1] 
        u_I[0       , 1] = u_d[0           , 1] 

        u_I[elements - 1, 0] = u_d[elements - 2, 1] 
        u_I[elements - 1, 1] = u_d[elements - 1, 1] 


        return u_I

    def dg(self, u, dt):
        order     = self.order
        Np        = self.Np
        elements  = self.elements

        intPoints = self.intPoints
        x_r       = self.x_r

        dg_l      = self.dg_l
        dg_r      = self.dg_r

        du        = self.getDiscontDeri(u)        # Get uncorrected derivative
        
        u_d       = self.getDiscontBnd(u )         # Get discontinuous values at element boundary

        u_I       = self.getInteractionValues(u_d)
        
        rhs       = np.zeros_like(u)

        for i in range(elements):
            rhs[i] = du[i]  
            rhs[i] = rhs[i] + (u_I[i, 0] - u_d[i, 0])*dg_l 
            rhs[i] = rhs[i] + (u_I[i, 1] - u_d[i, 1])*dg_r 
        
            rhs[i] = -1*rhs[i]/x_r[i]

        return rhs

    def ssp_rk43(self, dt, u, rhs_fn):
        rhs =  rhs_fn(u, dt)
        y   =  u + (1.0/2)*dt*rhs

        rhs =  rhs_fn(y, dt)
        y   =  y + (1.0/2)*dt*rhs
        
        rhs =  rhs_fn(y, dt)
        y   =  (2.0/3)*u  + (1.0/3)*y + (1/6)*dt*rhs

        rhs =  rhs_fn(y, dt)
        u   =  y + (1.0/2)*dt*rhs 

        return u 


    def euler(self, dt, u, rhs_fn):
        return u + dt*rhs_fn(u, dt)

    def ssp_rk33(self, dt, u, rhs_fn):
        # x1 = x + k0, t1 = t + dt, k1 = dt*f(t1, x1)
        # x2 = 3/4*x + 1/4*(x1 + k1), t2 = t + 1/2*dt, k2 = dt*f(t2, x2)
        # x3 = 1/3*x + 2/3*(x2 + k2), t3 = t + dt

        rhs =  rhs_fn(u, dt)
        y   =  u + dt*rhs

        rhs =  rhs_fn(y, dt)
        y   = (3.0/4)*u + (1.0/4)*y + (1.0/4)*dt*rhs
        
        rhs =  rhs_fn(y, dt)
        u   = (1.0/3)*u + (2.0/3)*y + (2.0/3)*dt*rhs

        return u 


    def wave_solver(self, dt, T_final):
        u   = self.u

        T   = 0
        dt_real = min(dt, T_final - T)

        it_coun = 0
        while (T < T_final) :
            u   = self.ssp_rk33(dt, u, self.dg) 

            T       = T + dt_real
            dt_real = min(dt, T_final - T)

            if (it_coun % 10 == 0):
                print('Time: ', T, ' Max u: ', np.max(u))

            it_coun  = it_coun + 1

        self.u = u
        
        self.plot(self.intPoints, u)

    def filter_van(self, u):
        order    = self.order
        Np       = self.Np

        elements = self.elements 

        filter_v = self.filterVan

        for i in range(elements):
            u[i] = np.dot(filter_v, u[i])


    def filtered_wave_solver(self, dt, T_final):
        u   = self.u
        self.filter_van(u)

        T   = 0
        dt_real = min(dt, T_final - T)

        it_coun = 0
        while (T < T_final) :
            u       = self.ssp_rk33(dt, u, self.dg) 
            self.filter_van(u)

            T       = T + dt_real
            dt_real = min(dt, T_final - T)

            if (it_coun % 10 == 0):
                print('Time: ', T, ' Max u: ', np.max(u))

            it_coun  = it_coun + 1

        self.u = u
        
        self.plot(self.intPoints, u)

    
    def error(self):
        order     = self.order
        Np        = self.Np

        elements  = self.elements 

        intPoints = self.intPoints

        ex        = np.zeros((self.elements, self.Np))

        self.project(self.fn, intPoints, ex) # Set exact condition

        l2_error  = np.linalg.norm(ex - self.u) 

        print("L2 error is ", l2_error/(Np*elements))

    def fn(self, x):
#        return 1.0 + 0.2*np.sin(np.pi*(x))
        return np.exp(-40*(x)**2) 


if __name__=="__main__":
    order    = 5
    elements = 25
    startX   = -1
    stopX    =  1
    
    '''
    correctType 
    0: Radau
    1: Krivodnova , takes correcFac
    '''
    run     = WaveDG(order, elements, startX, stopX, correcType = 1, correcFac = 1.0)

    dt      = 0.003  

    T_final = 8.0
#    run.wave_solver(dt, T_final)
    run.filtered_wave_solver(dt, T_final)

    run.error()


