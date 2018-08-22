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

        self.num_faces = elements + 1

        self.x_r, self.intPoints, self.f2e = self.dgMesh(self.elements, self.startX, self.stopX, self.order)

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

        # (showing 1 interior node)
        # Faces:
        #    0         1         2         3         4
        #    *----x----*----x----*----x----*----x----*
        #    Elements:
        #        (0)       (1)       (2)       (3)

        # Creating face connectivity

        num_faces = elements + 1

        f2e = np.zeros( (num_faces, 2) ) # 0: Left element, 1: Right element

        for i in range(1, num_faces - 1):
            f2e[i, 0] = i - 1
            f2e[i, 1] = i 

        # Periodic boundaries
        f2e[0,             0] = elements - 1 
        f2e[0,             1] = 0 

        f2e[num_faces - 1, 0] = elements - 1 
        f2e[num_faces - 1, 1] = 0 

        return(x_r, intPoints, f2e)


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


    def getFaceProj(self): 
        """
        Get face projections matrices
        P_L projects solution vector to left side of faces
        P_R projects solution vector to rght side of faces
        """
        var_dim   = self.var_dim

        elements  = self.elements 
        Np        = self.Np       

        num_faces = self.num_faces

        l_R       = self.l_R  
        l_L       = self.l_L 
        
        f2e       = self.f2e
        
        P_L       = np.zeros( (var_dim*num_faces, var_dim*elements*Np) )
        P_R       = np.zeros( (var_dim*num_faces, var_dim*elements*Np) )

        size = elements*Np
        for i in range(num_faces):
            left = f2e[i, 0]
            rght = f2e[i, 1]
            for vd in range(var_dim):
                P_L[vd*num_faces + i, size*vd + left*Np:size*vd + left*Np + Np] = l_L 
                P_R[vd*num_faces + i, size*vd + rght*Np:size*vd + rght*Np + Np] = l_R 

#        self.plot_coo_matrix(P_R)

        return P_L, P_R


    def getDF_du_l(self, u): # Get DF(u_L)/Du_L :Requires Jacobian matrix of u_L 
        var_dim   = self.var_dim
        elements  = self.elements 
        num_faces = self.num_faces
        Np        = self.Np      

        gamm      = self.gamm

        P_l, P_r  = self.getFaceProj()  
    
        u_l = np.dot(P_l, u) # Get face projections on the left side
        u_r = np.dot(P_r, u) # Get face projections on the rght side

        df_dul = np.zeros( (var_dim*num_faces, var_dim*num_faces) )
        df_dur = np.zeros( (var_dim*num_faces, var_dim*num_faces) )

#        for i in range(var_dim*num_faces):
#            print(u_l[i], u_r[i])

        u_f_l = np.zeros(var_dim)
        u_f_r = np.zeros(var_dim)
        for i in range(num_faces):
            for j in range(var_dim):
                u_f_l[j] = u_l[j*num_faces + i]
                u_f_r[j] = u_r[j*num_faces + i]
            j_p_l = self.getPointEulerJacobian(var_dim, gamm, u_f_l)
            j_p_r = self.getPointEulerJacobian(var_dim, gamm, u_f_r)
#            print(np.dot(j_p, u_f))

            for v1 in range(var_dim):
                for v2 in range(var_dim):
                    df_dul[v1*num_faces + i, v2*num_faces + i] = j_p_l[v1, v2] 
                    df_dur[v1*num_faces + i, v2*num_faces + i] = j_p_r[v1, v2] 

#            print(df_dul)

#        self.plot_coo_matrix(df_dul)

#        print(u_l)
        print(np.dot(df_dul, u_l))
        print(np.dot(df_dur, u_r))





    def euler_solver(self, dt, T_final):
        u   = self.u

        var_dim   = self.var_dim
        elements  = self.elements 
        Np        = self.Np       

        size      = elements*Np

        self.getEulerJacobian(var_dim, size, u)

        Dx   = self.getDerivativeMatrix()
        
        f  = self.getEulerFlux(var_dim, size, u)

#        print(u)
#        print(f)

        self.getDF_du_l(u) # Get DF/Du_L :Requires Jacobian matrix of u_L 



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


