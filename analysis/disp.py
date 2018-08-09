import sympy
import numpy as np

import matplotlib.pyplot as plt

class Poly:

    def lagrange_right(self, order):
        """
        Lagrange interpolation matrix at right boundary 
        """
        Np = order + 1
        nodes = np.polynomial.legendre.leggauss(Np)[0] 

        return self.lagrangeInterpolate(nodes,  1.0) 

    def lagrange_left(self, order):
        """
        Lagrange interpolation matrix at right boundary 
        """
        Np = order + 1
        nodes = np.polynomial.legendre.leggauss(Np)[0] 

        return self.lagrangeInterpolate(nodes, -1.0) 


    def lagrangeInterpolate(self, nodes, x):
        """
        Lagrange interpolation matrix at a particular point 
        """
        phi = self.lagrange(nodes)
        l_I = np.zeros((len(nodes)))
        
        r = sympy.Symbol('r')

        for i in range(len(nodes)):
            l_I[i] = phi[i].evalf(subs = {r: x})
        return l_I 

    def lagrange(self, nodes):
        """
        Lagrange polynomial 
        """
        length = len(nodes)
        r = sympy.Symbol('r')
        phi = sympy.ones(1, length)
        for k in range(length):
            for l in range(length):
                if (k != l):
                    phi[k] *= (r - nodes[l])/(nodes[k] - nodes[l])
        return phi 


    def lagrangeDeri(self, nodes):
        """
        Lagrange matrix at the nodes is just an Identity
        We'll come back to interpolation at points other than nodes
        at a later time
        Here we create derivative operator at the nodes
        Lagrange polynomial is
        phi = Product(l, l.neq.k) (r - r_l)/(r_k - r_l)
        r_i are the nodes
        """
        length = len(nodes)
        r = sympy.Symbol('r')
        phi = sympy.ones(1, length)
        dPhi = sympy.zeros(length, length)
        for k in range(length):
            for l in range(length):
                if (k != l):
                    phi[k] *= (r - nodes[l])/(nodes[k] - nodes[l])
        for k in range(length):
            for l in range(length):
                dPhi[k, l] = sympy.diff(phi[l]).evalf(subs = {r: nodes[k]})
        return dPhi

    def legendre(self, order):
        """
        Legendre Polynomials
        P_0 = 1
        P_1 = r
        (k+1)P_(k+1) = (2k+1)rP_k - kP_(k-1)
        """
        assert(int(order) == order) #Integer check
        r = sympy.Symbol('r')
        p_0 = 1
        p_1 = r
        if order == 0:
            return p_0
        if order == 1:
            return p_1
        else:
            for k in range(1, order):
                p_kP1 = ((2*k + 1)*r*p_1 - k*p_0)/(k+1)
                p_0 = p_1
                p_1 = p_kP1
        return sympy.simplify(p_kP1)

    def legendreDeri(self, order, nodes):
        """
        Return derivative of Legendre at nodes as a matrix operator
        Notice that the Legendre polynomial is the same polynomial
        at all points unlike Lagrange, so it becomes a Diagonal matrix
        """
        length = len(nodes)

        r = sympy.Symbol('r')
        p = self.legendre(order)

        legDPhi = sympy.zeros(length, length)

        for i in range(length):
            legDPhi[i, i] = sympy.diff(p).evalf(subs = {r: nodes[i]})
                
        return legDPhi

    def leftRadauDeri(self, order, nodes):
        """
        Return matrix of left Radau Deri operator
        g_l = ((-1)^k/2)*(L_k - L_(k+1))
        where L_k is the k order Legendre polynomial
        """
        length = len(nodes)

        r = sympy.Symbol('r')
        p = self.legendre(order)
        p1 = self.legendre(order+1)

        dg_l = np.zeros(length)

        for i in range(length):
            dg_l[i]  = sympy.diff(p).evalf(subs = {r: nodes[i]})
            dg_l[i] -= sympy.diff(p1).evalf(subs = {r: nodes[i]})
        dg_l *= (-1)**order/2
                
        return dg_l 
       
    def rightRadauDeri(self, order, nodes):
        """
        Return matrix of left Radau Deri operator
        g_r = ((L_k + L_(k+1)/2)
        where L_k is the k order Legendre polynomial
        """
        length = len(nodes)

        r = sympy.Symbol('r')
        p = self.legendre(order)
        p1 = self.legendre(order+1)

        dg_r = np.zeros(length)

        for i in range(length):
            dg_r[i]  = sympy.diff(p).evalf(subs = {r: nodes[i]})
            dg_r[i] += sympy.diff(p1).evalf(subs = {r: nodes[i]})
        dg_r *= 1/2
        
        return dg_r 


    def FReigvals(self, order):
        """
        We get eigenvalues for FR
        Based roughly on how Gassner, Kopriva did in in JSc 2011
        We also need to rearrage eigenvalues which are completely ad-hoc
        """
        Np = order + 1

        nodes    = np.polynomial.legendre.leggauss(Np)[0] 
        weights  = np.polynomial.legendre.leggauss(Np)[1] 

        D       = self.lagrangeDeri(nodes)

        H1      = np.zeros( (Np, Np) )
        H2      = np.zeros( (Np, Np) )
        DM      = np.zeros( (Np, Np) )

        r       = sympy.Symbol('r')
        l       = self.lagrange(nodes)

        dg_l    = self.leftRadauDeri(order,  nodes)
        dg_r    = self.rightRadauDeri(order, nodes)

        l_l     = self.lagrange_left(order)
        l_r     = self.lagrange_right(order)

        for i in range(Np):
            for j in range(Np):
                H1[i, j] = dg_l[i]*l_r[j] 
                H2[i, j] = dg_l[i]*l_l[j] 

                DM[i, j] = D[i, j]

        num     = 100
        kh      = np.linspace(0., Np*np.pi + 1E-2, num)

        eigvals = np.zeros( (num, Np), dtype = np.complex )

        for i in range(num):
            coeff = np.exp((0-1j)*kh[i])
            M     = DM + coeff*H1 - H2
    
            A     = (2./(0+1j))*M
   
            eigen = np.linalg.eig( A )

            eigvals[i] = eigen[0]

        realmodes = np.zeros( (num, Np), dtype = np.complex )
        imagmodes = np.zeros( (num, Np), dtype = np.complex )

        for j in range(Np):
            imagmodes[:, j] = -(0+1j)*(eigvals[:, j] - eigvals[:, j].real)
            realmodes[:, j] =  eigvals[:, j].real
        

        for pi_loop in range(1, Np + 1):
            pi_arg = np.argmin( np.abs(kh - pi_loop*np.pi)  ) # At which index kh is closes to pi
    
            if np.abs(kh[pi_arg] - pi_loop*np.pi) < 1E-5:
                pi_arg_r = pi_arg + 1
                pi_arg_l = pi_arg - 1
            elif kh[pi_arg] > np.pi :
                pi_arg_r = pi_arg 
                pi_arg_l = pi_arg - 1
            else :
                pi_arg_r = pi_arg + 1 
                pi_arg_l = pi_arg 

            eig_temp1 = realmodes[pi_arg_r, :].real
            eig_temp2 = realmodes[pi_arg_l, :].real
    
            diff      = np.zeros_like(eig_temp1) 
            min_diff  = np.zeros_like(eig_temp1) 
    
            for i in range(Np):
                diff     = np.abs(eig_temp1[i] - eig_temp2)
    
                min_diff[i] = np.argmin(diff)

            temp1 = np.copy(imagmodes)
            temp2 = np.copy(realmodes)
    
            for j in range(Np):
                imagmodes[pi_arg_r:, j] = temp1[pi_arg_r:, min_diff[j]]
                realmodes[pi_arg_r:, j] = temp2[pi_arg_r:, min_diff[j]].real
    

        for j in range(Np):
#            plt.plot(kh, imagmodes[:, j] )

            plt.plot(kh, realmodes[:, j] )

        plt.show()



if __name__=="__main__":
    run = Poly()

    order = 2
    Np    = order + 1

    run.FReigvals(order)

    gaussNodes = np.polynomial.legendre.leggauss(Np)[0] 
    dPhi       = run.lagrangeDeri(gaussNodes)

    l_R        = run.lagrange_right(order)
    l_L        = run.lagrange_left (order)



