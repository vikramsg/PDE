import matplotlib.pyplot as plt
import numpy as np
import cmath as cmh

spacing  = 0.01 

def plotting_2D():
    X = np.arange(-6, 5, spacing)
    Y = np.arange(-6, 5, spacing)

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

    Z3    = X + Y*complex(0, 1) 
    alpha = 1.0
    #Markakis-Heun 
    G3    = 1 + Z3 + (1./2.)*Z3**2 + alpha*(1./12.)*Z3**3

    Z3 = abs(G3)

    Z4 = X + Y*complex(0, 1) 
    #RK3
    G4 = 1 + Z4 + (1./2.)*Z4**2 + (1./6.)*Z4**3 

    Z4 = abs(G4)

    Z5 = X + Y*complex(0, 1) 
    #SSPRK34
    G5 = 1 + Z5 + (1./2.)*Z5**2 + (1./6.)*Z5**3 + (1./48)*Z5**4 

    Z5 = abs(G5)

    Z6 = X + Y*complex(0, 1) 
    #test
    G6 = 1 + Z6 + (1./2.)*Z6**2 + (1./6.)*Z6**3 + (1./24)*Z6**4 + (1./224)*Z6**5 

    Z6 = abs(G6)

#    C1 = plt.contour(X, Y, Z1,  np.arange(-1.01, 1.01, 0.1), colors = 'y')
#    C2 = plt.contour(X, Y, Z2,  np.arange(-1.01, 1.01, 0.1), colors = 'b')
    C3 = plt.contour(X, Y, Z3,  np.arange(-1.01, 1.01, 0.1), colors = 'g')
#    C4 = plt.contour(X, Y, Z4,  np.arange(-1.01, 1.01, 0.1), colors = 'r')
#    C5 = plt.contour(X, Y, Z5,  np.arange(-1.01, 1.01, 0.1), colors = 'm')
    C6 = plt.contour(X, Y, Z6,  np.arange(-1.01, 1.01, 0.1), colors = 'm')

#    h1,_ = C2.legend_elements()
#    h2,_ = C3.legend_elements()
#    h3,_ = C4.legend_elements()
#
#    plt.legend([h1[0], h2[0], h3[0]], ['Heun', 'New', 'RK4'])

    plt.show()


plotting_2D()


