import numpy as np
from scipy.sparse import coo_matrix
from polynomial import *
import matplotlib.pyplot as plt

class EulerDG:

    def __init__(self, order, elements, startX, stopX):
        ###########
        # Constants
        ##########

        self.gamm     = 1.4
        self.R        = 287.
        
        ##########

        self.order    = order
        self.Np       = order + 1

        self.var_dim  = 3

        self.elements = elements 
        self.startX   = startX 
        self.stopX    = stopX

        self.x_r, self.intPoints = self.dgMesh(self.elements, self.startX, self.stopX, self.order)

        self.u   = np.zeros( self.var_dim*self.elements*self.Np )
        
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
        var_dim  = self.var_dim

        size     = elements*Np

        for i in range(elements):
            for j in range(Np):
                init_sol = fn(x[i, j])
                for k in range(var_dim):
                    u[k*size + i*Np + j] = init_sol[k]

    def fn(self, x):
        u    = np.zeros(3)

        rho  = 1.0 + 0.2*np.sin(np.pi*(x))
        p    = 1
        vel  = 1

        u[0] = rho
        u[1] = rho*vel
        u[2] = p/(self.gamm - 1) + 0.5*rho*vel*vel

        return u 


    def getPointEulerFlux(self, u):
        f = np.zeros_like(u)

        rho   = u[0]
        rhoV  = u[1]
        rho_e = u[2]

        v      = rhoV/rho
        v_sq   = v*v

        p      = (self.gamm - 1)*(rho_e - 0.5*rho*v_sq)

        f[0]   = rhoV
        f[1]   = rho*v_sq + p
        f[2]   = (rho_e + p)*v 

        return f

    def getEulerFlux(self, var_dim, size, u):
        '''
        Get the 1D euler fluxes for the given solution vector
        '''
        f        = np.zeros_like(u)
      
        for i in range(size):
            rho   = u[         i]
            rhoV  = u[  size + i]
            rho_e = u[2*size + i]
    
            v      = rhoV/rho
            v_sq   = v*v
    
            p      = (self.gamm - 1)*(rho_e - 0.5*rho*v_sq)
    
            f[         i]   = rhoV
            f[  size + i]   = rho*v_sq + p
            f[2*size + i]   = (rho_e + p)*v 


        return f
    
                
    def getPointEulerJacobian(self, var_dim, gamm, u):
        J    = np.zeros( (var_dim, var_dim) )

        rho    = u[0]
        rhoV   = u[1]
        rho_e  = u[2]

        E      = rho_e

        v      = rhoV/rho
        v_sq   = v*v

        p      = (self.gamm - 1)*(rho_e - 0.5*rho*v_sq)

        a      = np.sqrt(gamm*p/rho)

        J[0, :] = [0, 1, 0]

        J[1, 0] = -0.5* (rhoV**2/rho**2) * (3 - gamm)
        J[1, 1] =       (rhoV   /rho   ) * (3 - gamm)
        J[1, 2] =                       (gamm - 1)
        
        J[2, 0] = -( (rhoV * E *gamm)/rho**2 ) +       (rhoV**3/rho**3)*(gamm - 1)
        J[2, 1] =  ( (       E *gamm)/rho    ) - 1.5 * (rhoV**2/rho**2)*(gamm - 1)
        J[2, 2] =  ( (    rhoV *gamm)/rho    ) 

        return J

    
    def getEulerJacobian(self, var_dim, size, u):
        '''
        Get the 1D euler fluxes for the given solution vector
        '''

        J = np.zeros( (var_dim, var_dim) )

        cons = np.zeros(var_dim)
        for i in range(size):
            cons[0] = u[         i]
            cons[1] = u[  size + i]
            cons[2] = u[2*size + i]
            
            J      = self.getPointEulerJacobian(var_dim, self.gamm, cons)

#                f      = self.getPointEulerFlux(u[:, i, j])

#                print(np.dot(J, u[:, i, j]), f)



    
    def getDerivativeMatrix(self):
        var_dim   = self.var_dim

        elements  = self.elements 
        Np        = self.Np       

        dPhi      = np.asarray(self.dPhi)

        Dx        = np.zeros( (var_dim*elements*Np, var_dim*elements*Np) )

        size = elements*Np
        for vd in range(var_dim):
            for i in range(elements):
                for j in range(Np):
                    Dx[vd*size + i*Np + j, vd*size + i*Np:vd*size + i*Np + Np] = dPhi[j, :]

#        self.plot_coo_matrix(Dx)

        return Dx


    def getLeftFaceProj(self): 
        var_dim   = self.var_dim

        elements  = self.elements 
        Np        = self.Np       

        l_R       = self.l_R  
        l_L       = self.l_L 
        
        P_L      = np.zeros( (var_dim*elements, var_dim*elements*Np) )

        size = elements*Np
        for i in range(elements):
            for vd in range(var_dim):
                P_L[i*var_dim + vd, (i*var_dim + vd)*Np:(i*var_dim + vd + 1)*Np] = l_L 

#        self.plot_coo_matrix(DF_L)

        return P_L

    def getRghtFaceProj(self): 
        var_dim   = self.var_dim

        elements  = self.elements 
        Np        = self.Np       

        l_R       = self.l_R  
        l_L       = self.l_L 
        
        P_R      = np.zeros( (var_dim*elements, var_dim*elements*Np) )

        size = elements*Np
        for i in range(elements):
            for vd in range(var_dim):
                P_R[i*var_dim + vd, (i*var_dim + vd)*Np:(i*var_dim + vd + 1)*Np] = l_R 

        return P_R




    def euler_solver(self, dt, T_final):
        u   = self.u

        var_dim   = self.var_dim
        elements  = self.elements 
        Np        = self.Np       

        size      = elements*Np

        self.getEulerJacobian(var_dim, size, u)

        Dx   = self.getDerivativeMatrix()
        
        P_l = self.getLeftFaceProj()  # Get Projection matrix of projecting onto left face of element
        P_r = self.getRghtFaceProj()  # Get Projection matrix of projecting onto rght face of element
    
        f  = self.getEulerFlux(var_dim, size, u)

#        print(u)
#        print(f)
#        print(np.dot(P_r, f))



    def plot_coo_matrix(self, m):
        m = coo_matrix(m)

        fig = plt.figure()

        ax = fig.add_subplot(111, axisbg='black')
        ax.plot(m.col, m.row, 's', color='white', ms=10)

        ax.set_xlim(0, m.shape[1])
        ax.set_ylim(0, m.shape[0])
        ax.set_aspect('equal')

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.invert_yaxis()

        ax.set_xticks([])
        ax.set_yticks([])
   
        plt.show()
 

if __name__=="__main__":
    order    = 2
    elements = 3 
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


