import sympy
import numpy as np

class Poly:

    def lagrange_right(self, order):
        """
        Lagrange interpolation matrix at right boundary 
        """
        nodes = self.gaussNodes(order)
        return self.lagrangeInterpolate(nodes,  1.0) 

    def lagrange_left(self, order):
        """
        Lagrange interpolation matrix at right boundary 
        """
        nodes = self.gaussNodes(order)
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


    def gaussNodes(self, order):
        """
        /////////////////////
        All nodes should be defined in the domain [-1, 1]
        ////////////////////
        Return Gauss nodes
        """
        assert(order) > 0
        numNodes = order + 1

        nodes = np.polynomial.legendre.leggauss(numNodes)[0] 

        return nodes
    
    def vandermonde(self, order, nodes):
        '''
        Create Vandermonde matrix to convert from Lagrange to Legendre
        '''
        r = sympy.Symbol('r')

        Np  = order + 1
        van = np.zeros((Np, Np))

        for i in range(Np):
            for j in range(Np):
                gamma     = 2.0/(2*j + 1)
                leg       = self.legendre(j)/np.sqrt(gamma)
                if isinstance(leg, float) :
                    van[i, j] = leg 
                else:
                    van[i, j] = leg.evalf(subs = {r: nodes[i]})
        return van

    
    def lagToLeg(self, order, nodes, van):
        '''
        Convert Lagrange poly to normalized Legendre
        '''
        r   = sympy.Symbol('r')
        phi = self.lagrange(nodes)

        Np  = order + 1
        P   = sympy.zeros(1, Np)

        # Leg from Lag
        for i in range(Np):
            for j in range(Np):
                P[i] = P[i] + van.T[i, j] * phi[j] # V^T . l

        Prin = False 
        if Prin:
            for i in range(Np):
                gamma     = 2.0/(2*i + 1)
                leg       = self.legendre(i)/np.sqrt(gamma)
    
                print((P[i] - leg).simplify()) # Should be zero
                print(i, P[i].evalf(subs = {r: 1}), lag[i].evalf(subs = {r: 1}))

        return P

    def modLagToLeg(self, order, nodes, van):
        '''
        Convert Lagrange poly to normalized Legendre
        '''
        r   = sympy.Symbol('r')
        phi = self.lagrange(nodes)

        Np  = order + 1
        P   = sympy.zeros(1, Np)

        # Leg from Lag
        for i in range(Np):
            for j in range(Np):
                P[i] = P[i] + van.T[i, j] * phi[j] # V^T . l

        # Lag from Leg
        inv_vanT = np.linalg.inv(van.T)
        lag = sympy.zeros(1, Np)
        for i in range(Np):
            for j in range(Np):
                lag[i] = lag[i] + inv_vanT[i, j] * P[j] # (V^T)^-1 . P
                
            print(i, lag[i].evalf(subs = {r: 1}))


        Prin = False 
        if Prin:
            for i in range(Np):
                gamma     = 2.0/(2*i + 1)
                leg       = self.legendre(i)/np.sqrt(gamma)
    
                print((P[i] - leg).simplify()) # Should be zero
                print(i, P[i].evalf(subs = {r: 1}), lag[i].evalf(subs = {r: 1}))

        return P




if __name__=="__main__":
    run = Poly()

    order = 1
    nodes = run.gaussNodes(order)

    van   = run.vandermonde(order, nodes)
    run.modLagToLeg(order, nodes, van)

#    phi  = run.lagrange(nodes)
#    print(phi)
#    l_I  = run.lagrangeInterpolate(nodes, -1)
#    print(l_I)

#    print(nodes)
#    dPhi  = run.lagrangeDeri(nodes)
#    print(dPhi)
#    import numpy.linalg as nplg
#    a = np.zeros((order + 1, order + 1))
#    for i in range(order + 1):
#        for j in range(order + 1):
#            a[i, j] = float(dPhi[i, j])
#    print(nplg.eig(a))
#
#    r = sympy.Symbol('r')
#    for order in range(1, 4):
#        print(order)
#        leg = run.legendre(order)
#        print(leg)
#        print(leg.evalf(subs = {r: 0.0}))
#

#    nodes = np.zeros(2)
#    nodes[0] = -0.577350
#    nodes[1] =  0.577350
#    dPhi = run.lagrangeDeri(nodes)
#    print(dPhi)
#    legDPhi = run.legendreDeri(1, nodes)
#    print(legDPhi)
#    dg_l = run.leftRadauDeri(1, nodes)
#    dg_r = run.rightRadauDeri(1, nodes)
#    print(dg_r)

    
#    nodes = np.zeros(3)
#    nodes[0] = -0.774597
#    nodes[1] =  0 
#    nodes[2] =  0.774597
#    dPhi = run.lagrangeDeri(nodes)
#    print(dPhi)
#    legDPhi = run.legendreDeri(2, nodes)
#    print(legDPhi)
#    dg_l = run.leftRadauDeri(2, nodes)
#    dg_r = run.rightRadauDeri(2, nodes)


#    
#    nodes = np.zeros(4)
#    nodes[0] = -0.861136
#    nodes[1] = -0.339981
#    nodes[2] =  0.339981
#    nodes[3] =  0.861136
#    dPhi = run.lagrangeDeri(nodes)
#    print(dPhi)
#    legDPhi = run.legendreDeri(3, nodes)
#    print(legDPhi)

