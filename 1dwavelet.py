import numpy as np


class Cells:

    def __init__(self):
        self.startX = None
        self.stopX = None
        self.elemNo = None



    def cellDefinition(self, elemNo, startX, stopX):
        self.elemNo = elemNo
        self.startX = startX
        self.stopX = stopX


class Grid:

    def __init__(self, noElements):
        self.noElements = noElements

        self.cells = [None] * noElements

        self.u = np.zeros((noElements, 3), dtype = float)

    def initMesh(self, x, connectivity):
        for i in range(self.noElements):
            self.cells[i] = Cells()
            self.cells[i].cellDefinition(i, x[i], x[i+1])
        self.connectivity = connectivity



class Solver:

    def __init__(self):
        return

    def initSolver(self, grid):
        self.grid = grid






def initSol(cells, u, gamma):
    """
    Initialize variables
    """
    for i, cell_i in enumerate(cells):
        x_i = 0.5*(cell_i.startX+cell_i.stopX)
        if (x_i < 0):
            u[i, 0] = 1
            u[i, 1] = 0
            u[i, 2] = 100000/(gamma-1) 
        else:
            u[i, 0] = 0.125
            u[i, 1] = 0
            u[i, 2] = 10000/(gamma-1)
    return u



if __name__=="__main__":
    xMin = -10
    xMax =  10
    nx = 1001

    x = np.linspace(xMin,xMax, nx)

    """
    Connectivity data. 0 index has left cell, 1 has right cell
    """
    connectivity = np.zeros((nx-1, 2), dtype = float)

    for i in range(1, nx-2):
        connectivity[i, 0] = i-1
        connectivity[i, 1] = i+1
    
    connectivity[0, 0] = -1
    connectivity[0, 1] = 1
    connectivity[-1, 0] = nx-3 
    connectivity[-1, 1] = -1

    grid = Grid(noElements = nx-1)
    grid.initMesh(x, connectivity)

    gamma = 1.4
    initSol(grid.cells, grid.u, gamma)

    run = Solver() 
    run.initSolver(grid)
