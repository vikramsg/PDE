import matplotlib.pyplot as plt
import numpy as np
import cmath as cmh

spacing  = 0.01 

def plotting_2D():
    X = np.arange(-5, 4, spacing)
    Y = np.arange(-5, 4, spacing)

    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()

    Z1 = X + Y*complex(0, 1) 
    #Euler
    G1 = 1 + Z1;

    Z1 = abs(G1)

    Z2 = X + Y*complex(0, 1) 
    # Heun's modified Euler
    G2 = 1 + Z2 + 0.5*Z2**2;

    Z2 = abs(G2)

    Z3 = X + Y*complex(0, 1) 
    #Markakis-Heun 
    G3 = 1 + Z3 + (1./2.)*Z3**2 + (1./12.)*Z3**3

    Z3 = abs(G3)

    Z4 = X + Y*complex(0, 1) 
    #RK4
    G4 = 1 + Z4 + (1./2.)*Z4**2 + (1./6.)*Z4**3 + (1./24.)*Z4**4

    Z4 = abs(G4)

#    C1 = plt.contour(X, Y, Z1,  np.arange(-1.01, 1.01, 0.1), colors = 'y')
    C2 = plt.contour(X, Y, Z2,  np.arange(-1.01, 1.01, 0.1), colors = 'b')
    C3 = plt.contour(X, Y, Z3,  np.arange(-1.01, 1.01, 0.1), colors = 'g')
    C4 = plt.contour(X, Y, Z4,  np.arange(-1.01, 1.01, 0.1), colors = 'r')

#    plt.clabel(C2, inline=1, fontsize=10)
    
    plt.show()


plotting_2D()


