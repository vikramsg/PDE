import numpy as np
import scipy 
from polynomial import *
import matplotlib.pyplot as plt

class EulerDG:

    def __init__(self, order, elements, startX, stopX):
        ###########
        # Constants
        ##########

        self.gamm     = 1.4
        
        ##########


        self.order    = order
        self.Np       = order + 1

        self.var_dim  = 3

        self.elements = elements 
        self.startX   = startX 
        self.stopX    = stopX

        self.x_r, self.intPoints = self.dgMesh(self.elements, self.startX, self.stopX, self.order)

        self.u   = np.zeros((self.var_dim, self.elements, self.Np))
        self.rhs = np.zeros((self.var_dim, self.elements, self.Np))
        
        self.project(self.fn, self.intPoints, self.u) # Set initial condition

        self.nodes = np.polynomial.legendre.leggauss(self.Np)[0] 

        self.dg_l  = Poly().leftRadauDeri(order, self.nodes)
        self.dg_r  = Poly().rightRadauDeri(order, self.nodes)

        gaussNodes = np.polynomial.legendre.leggauss(self.Np)[0] 
        self.dPhi  = Poly().lagrangeDeri(gaussNodes)

        self.l_R   = Poly().lagrange_right(order)
        self.l_L   = Poly().lagrange_left (order)


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
                u[:, i, j] = fn(x[i, j])

    def fn(self, x):
        u    = np.zeros(3)

        rho  = 1.0 + 0.2*np.sin(np.pi*(x))
        p    = 1
        vel  = 1

        u[0] = rho
        u[1] = rho*vel
        u[2] = p/(self.gamm - 1) + 0.5*rho*vel*vel

        return u 


    def getDiscontDeri(self, u):
        var_dim  = u.shape[0] #variable dimension of equation (=3 for Euler equations)
        elements = u.shape[1] #Num of elements
        Np       = u.shape[2] #Num of soln points

        du       = np.zeros_like(u)

        dPhi     = self.dPhi 

        for i in range(var_dim):
            for j in range(elements):
                du[i, j, :] = np.dot(np.array(dPhi), np.array(u[i, j, :]))

        return(du)

   
    def getDiscontBnd(self, u):
        var_dim  = u.shape[0] #variable dimension of equation (=3 for Euler equations)
        elements = u.shape[1] #Num of elements
        Np       = u.shape[2] #Num of soln points
      
        order    = Np - 1 # Only true in 1D

        l_R      = self.l_R 
        l_L      = self.l_L 

        u_d = np.zeros((var_dim, elements, 2))

        for i in range(var_dim):
            for j in range(elements):
                u_d[i, j, 0] = np.dot(l_L, u[i, j, :])
                u_d[i, j, 1] = np.dot(l_R, u[i, j, :])

        return(u_d)


    def getInteractionValues(self, u_d, f_d):
        var_dim  = u_d.shape[0] #variable dimension of equation (=3 for Euler equations)
        elements = u_d.shape[1] #Num of elements

        f_I = np.zeros((var_dim, elements, 2))

        # Pure upwinding
        for i in range(var_dim):
            for j in range(1, elements - 1):
                f_I[i, j, 0] = f_d[i , j - 1, 1] 
                f_I[i, j, 1] = f_d[i , j    , 1] 

        # Periodic boundary
        for i in range(var_dim):
            f_I[i,   0  , 0] = f_d[i, elements - 1, 1] 
            f_I[i,   0  , 1] = f_d[i, 0           , 1] 

            f_I[i, elements - 1, 0] = f_d[i, elements - 2, 1] 
            f_I[i, elements - 1, 1] = f_d[i, elements - 1, 1] 

        return f_I



    def dg(self, u, dt):
        order     = self.order
        Np        = self.Np
        elements  = self.elements
        var_dim   = self.var_dim

        intPoints = self.intPoints
        x_r       = self.x_r

        dg_l      = self.dg_l
        dg_r      = self.dg_r

        f_E       = self.getEulerFlux(u)         # Get Euler flux 

        df        = self.getDiscontDeri(f_E)     # Get uncorrected derivative
        
        u_edge    = self.getDiscontBnd(u   )     # Get discontinuous values at element boundary
        f_edge    = self.getDiscontBnd(f_E )     # Get discontinuous values at element boundary

        f_I       = self.getInteractionValues(u_edge, f_edge)
        
        rhs       = np.zeros_like(u)

        for i in range(var_dim):
            for j in range(elements):
                rhs[i, j] = df[i, j]  
                rhs[i, j] = rhs[i, j] + (f_I[i, j, 0] - f_edge[i, j, 0])*dg_l 
                rhs[i, j] = rhs[i, j] + (f_I[i, j, 1] - f_edge[i, j, 1])*dg_r 
            
                rhs[i, j] = -1*rhs[i, j]/x_r[j]

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


        
    def getEulerFlux(self, u):
        '''
        Get the 1D euler fluxes for the given solution vector
        '''
        var_dim  = u.shape[0] #variable dimension of equation (=3 for Euler equations)
        elements = u.shape[1] #Num of elements
        Np       = u.shape[2] #Num of soln points

        f        = np.zeros_like(u)
      
        for i in range(elements):
            for j in range(Np):
                rho   = u[0, i, j]
                rhoV  = u[1, i, j]
                rho_e = u[2, i, j]

                v      = rhoV/rho
                v_sq   = v*v

                p      = (self.gamm - 1)*(rho_e - 0.5*rho*v_sq)

                f[0, i, j] = rhoV
                f[1, i, j] = rho*v_sq + p
                f[2, i, j] = (rho_e + p)*v 

        return f

    

    def euler_solver(self, dt, T_final):
        u   = self.u

        T   = 0
        dt_real = min(dt, T_final - T)

        it_coun = 0
        while (T < T_final) :
            u   = self.ssp_rk43(dt, u, self.dg) 

            T       = T + dt_real
            dt_real = min(dt, T_final - T)

            if (it_coun % 10 == 0):
                print('Time: ', T, ' Max u: ', np.max(u))

            it_coun  = it_coun + 1

        self.u = u

        self.plot(self.intPoints, u, 0)
 
    def plot(self, x, u, vDim):
        '''
        vDim is the particular dimension that will be plotted
        '''
        var_dim  = u.shape[0] #variable dimension of equation (=3 for Euler equations)
        elements = u.shape[1] #Num of elements
        Np       = u.shape[2] #Num of soln points

        size     = elements*Np

        xv       = np.zeros((size))
        uv       = np.zeros((size))

        coun = 0
        for i in range(elements):
            for j in range(Np):
                xv[coun] = x[      i, j]
                uv[coun] = u[vDim, i, j]
                coun     = coun + 1

        plt.plot(xv, uv)

        plt.show(       )
 

if __name__=="__main__":
    order    = 3
    elements = 10
    startX   = -1
    stopX    =  1
    
    '''
    correctType 
    0: Radau
    1: Krivodnova , takes correcFac
    '''
    run     = EulerDG(order, elements, startX, stopX)

    dt      = 0.002

    T_final = 0.25
    run.euler_solver(dt, T_final)


