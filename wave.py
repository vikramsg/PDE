import numpy as np
import scipy 
from polynomial import *
import matplotlib.pyplot as plt

class WaveDG:

    def __init__(self, order, elements, startX, stopX, time_stepping = 0,
                    correcType = 0, correcFac = 0.9):
        self.order    = order
        self.Np       = order + 1

        self.ts       = time_stepping

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


    def getCentralFlux(self, u_d):
        elements = u_d.shape[0]
        Np       = u_d.shape[1]

        u_I = np.zeros((elements, 2))

        alpha = 0.5

        #Central flux
        for i in range(1, elements - 1):
            u_I[i, 0] = alpha*(u_d[i - 1, 1] + u_d[i    , 0] )
            u_I[i, 1] = alpha*(u_d[i    , 1] + u_d[i + 1, 0] )

        # Periodic boundary
        u_I[0       , 0] = alpha*(u_d[elements - 1, 1] + u_d[0    , 0] )
        u_I[0       , 1] = alpha*(u_d[0           , 1] + u_d[0 + 1, 0] )

        u_I[elements - 1, 0] = alpha*(u_d[elements - 2, 1] + u_d[elements - 1, 0] )
        u_I[elements - 1, 1] = alpha*(u_d[elements - 1, 1] + u_d[0           , 0] )

        return u_I


    def getUpwindFlux(self, u_d):
        elements = u_d.shape[0]
        Np       = u_d.shape[1]

        u_I = np.zeros((elements, 2))

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
        rhs      = self.dg_1st(u, 0) 
        rhs     *= -1.;

        return rhs


    def dg_2nd(self, u):
        '''
        Get second derivative using BR1
        '''
        q    = self.dg_1st(u, 0) 
        rhs  = self.dg_1st(q, 0) 

        return rhs


    def dg_1st(self, u, fluxType = 0):
        '''
        Calculate first derivative 
        fluxType = 0: Upwind
        fluxType = 1: Central 
        '''
        order     = self.order
        Np        = self.Np
        elements  = self.elements

        intPoints = self.intPoints
        x_r       = self.x_r

        dg_l      = self.dg_l
        dg_r      = self.dg_r

        du        = self.getDiscontDeri(u)        # Get uncorrected derivative
        
        u_d       = self.getDiscontBnd(u )         # Get discontinuous values at element boundary

        if (fluxType == 0):
            u_I       = self.getUpwindFlux(u_d)
        elif (fluxType == 1):
            u_I       = self.getCentralFlux(u_d)
        
        deri      = np.zeros_like(u)

        for i in range(elements):
            deri[i] = du[i]  
            deri[i] = deri[i] + (u_I[i, 0] - u_d[i, 0])*dg_l 
            deri[i] = deri[i] + (u_I[i, 1] - u_d[i, 1])*dg_r 
        
            deri[i] = deri[i]/x_r[i]

        return deri 


    def stabilized_2s(self, dt, u):
        # Stabilized 2 stage method 

        u0    = u

        rhs1  =  self.dg_1st(u, 0)
        rhs1 *= -1.;
        y     =  u + dt*rhs1

        d1    =  -1.*self.dg_2nd(u) # Second derivative of u_0
        d2    =  -1.*self.dg_2nd(y) # Second derivative of u_1

        rhs2  =  self.dg_1st(y, 0)
        rhs2 *= -1.;
        u     =  u + 0.5*dt*rhs1 + 0.5*dt*rhs2 + (dt**2/12.)*(d1 - d2)

        elements = u.shape[0]
        Np       = u.shape[1]

        for i in range(elements):
            u0l  = u0[i]
            ul   = u[i]
            f1   = rhs1[i]
            f2   = rhs2[i]
            eps1 = np.zeros_like(ul)

            en_norm = np.dot(ul, ul)/np.dot(u0l, u0l)

            den  = np.dot(u0l, u0l)
            if (den > 1E-14):
                eps1 = ( 0.25*dt**2*np.dot(f1 - f2, f1 - f2) )/np.dot(u0l, u0l)

            if en_norm > 1:
                u[i]   =  u0[i] + ( 
                            0.5*dt*rhs1[i] + 0.5*dt*rhs2[i] + (dt**2/12.)*(d1[i] - d2[i]) )/np.sqrt(
                                    1 + eps1)

        return u 



    def stabilized_23s(self, dt, u):
        # Stabilized 2 stage method third order

        rhs1  =  self.dg_1st(u, 0)
        rhs1 *= -1.;

        d1    =  -1.*self.dg_2nd(u) # Second derivative of u_0

        y     =  u + dt*rhs1 - ( dt**2/3. )*d1

        d2    =  -1.*self.dg_2nd(y) # Second derivative of u_1

        rhs2  =  self.dg_1st(y, 0)
        rhs2 *= -1.;

        u     =  u + 0.5*dt*rhs1 + 0.5*dt*rhs2 + (1./14.)*( (dt**2)*(d1 - d2) )

        return u 


    def rk22(self, dt, u, rhs_fn):
        # Heun's method 

        rhs1 =  rhs_fn(u, dt)
        y    =  u + dt*rhs1

        rhs2 =  rhs_fn(y, dt)
        u    =  u + 0.5*dt*rhs1 + 0.5*dt*rhs2
        
        unf  = np.ndarray.flatten(u)
        print("dt is ", dt , " energy is ", np.dot( unf, unf ))
        
        return u 

    def adapt_rk22(self, dt, u, rhs_fn):
        # Heun's method 

        dt_0  =  dt

        b1    = np.array( [1.0, 0.0] )
        b2    = np.array( [0.5, 0.5] )
        w     = np.array( [0., 2.] )

        eta1 =  u
        f1   =  rhs_fn(u, dt_0)

        flag = 0
        while flag == 0:
            eta2  =  u + dt_0*f1
            f2    =  rhs_fn(eta2, dt)

            eta1f = np.ndarray.flatten(eta1)
            f1f   = np.ndarray.flatten(f1)

            eta2f = np.ndarray.flatten(eta2)
            f2f   = np.ndarray.flatten(f2)

            num   = b2[0]*np.dot( eta1f, f1f ) + b2[1]*np.dot( eta2f, f2f )
            denom = w[1]*b2[1]*np.sqrt( np.dot( eta2f, eta2f ) )
    
            un    =  u + 0.5*dt_0*f1 + 0.5*dt_0*f2
            vn    =  u + 0.5*dt_0

            unf   = np.ndarray.flatten(un)
            vnf   = np.ndarray.flatten(vn)

            lhs   = np.sqrt( np.dot(unf - vnf, unf - vnf) )
            rhs   = dt_0*np.abs( num/denom )
            
            E     = ( b1[0] - b2[0] )*f1 + ( b1[1] - b2[1] )*f2 
            Ef    = np.ndarray.flatten(E)

            check = np.dot(eta2f, Ef)

            print(dt_0, lhs, rhs, check)

            if (lhs > rhs) and ( rhs > 1E-6):
                dt_0 *= 0.5
            else:
                flag = 1

        u = un

        print("dt init is ", dt ,"dt is ", dt_0, " energy is ", np.dot( unf, unf ))
#        print("dt init is ", dt ,"dt is ", dt_0, " energy is ", np.dot( unf, unf ))

        return u, dt_0


    def stab_rk22(self, dt, u, rhs_fn):
        # Heun's method

        u0   = u

        rhs1 =  rhs_fn(u, dt)
        y    =  u + dt*rhs1

        rhs2 =  rhs_fn(y, dt)
        u    =  u + 0.5*dt*rhs1 + 0.5*dt*( rhs2 )

        elements = u.shape[0]
        Np       = u.shape[1]

        for i in range(elements):
            u0l  = u0[i]
            ul   = u[i]
            f1   = rhs1[i]
            f2   = rhs2[i]
            eps1 = np.zeros_like(ul)

            en_norm = np.dot(ul, ul)/np.dot(u0l, u0l)

            den  = np.dot(u0l, u0l)
            if (den > 1E-14):
                eps1 = ( 0.25*dt**2*np.dot(f1 - f2, f1 - f2) )/np.dot(u0l, u0l)

            if en_norm > 1:
                u[i] = u[i]/np.sqrt(1 + eps1)

        return u 


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
        u       = self.u

        T       = 0
        dt_real = min(dt, T_final - T)
        dt_act  = dt_real 

        it_coun = 0

        u, dt_act   = self.adapt_rk22(dt_real, u, self.dg) 

#        while (T < T_final) :
#            if self.ts   == 0:
#                u           = self.rk22(dt_real, u, self.dg) 
#            elif self.ts == 1:
#                u, dt_act   = self.adapt_rk22(dt_real, u, self.dg) 
#            else:
#                raise NotImplementedError
#
#            T       = T + dt_act
#            dt_real = min(dt, T_final - T)
#            dt_act  = dt_real 
#
#            uflat   = np.ndarray.flatten(u) 
#
##            if (it_coun % 10 == 0):
##                print('Time: ', T, ' Max u: ', np.max(u), ' Energy: ',  np.dot(uflat, uflat) )
#
#            it_coun  = it_coun + 1
#
#        self.u = u
#
#        print(" Iterations is ", it_coun)
#        
#        self.plot(self.intPoints, u)

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
        return 1 + np.exp(-10*(x)**2) 


if __name__=="__main__":
    order    = 3
    elements = 25
    startX   = -1.
    stopX    =  1.

    CFL      =  0.9 
    
    '''
    correctType 
    0: Radau
    1: Krivodnova , takes correcFac
    '''
    run     = WaveDG(order, elements, startX, stopX, 
                        time_stepping = 1,
                        correcType = 0, correcFac = 1.0)

    h       = (stopX - startX)/elements

    dt      = CFL*h/(2.*order + 1.)

    T_final = 2.0  
    run.wave_solver(dt, T_final)
#    run.filtered_wave_solver(dt, T_final)

    run.error()


