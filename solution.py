import numpy as np


class Element:
    """
    NOTE: Efficiency considerations require that
    we use structure of arrays instead of array of
    structures. We'll need to sort it out at some point
    Maybe with trees to allow adaptation

    Create a solution container for each element
    u is the solution variable
    f is the flux variable
    """
    def __init__(self, numNodes):
        """
        Each element will be parameterized by 
        number of nodes inside
        Flux points are the solution points and 
        two boundary points
        """
        self.u = np.zeros(numNodes) 
        self.u_x = np.zeros(numNodes) 
        self.f = np.zeros(numNodes + 2)
        self.f_x = np.zeros(numNodes + 2)
        self.nodes = np.zeros(numNodes)

class Grid:

    def getGrid(self):
        """
        We are going to mimic a grid generator
        Create a 1D equally spaced grid
        numEle number of elements to create
        from startX to stopX
        gr will contain startX, stopX for each element
        and a third index to store its index in array
        This merely mimics connectivity
        """
        numEle = 9 
        startX = 0
        stopX = 1
        assert(stopX > startX)
        assert(numEle > 0)
        gr = np.zeros((numEle, 3))
        dx = (stopX - startX)/(numEle)
        for i in range(numEle):
            temp = startX + i*dx
            gr[i, 0] = temp
            gr[i, 1] = temp+dx
            gr[i, 2] = i
        return gr

class Field:

    def solution(self, numEle, order):
        """
        mesh will be the solution array 
        Each mesh instance will be of type Element
        numEle gives total number of elements
        order is the order of solution in each element
        """
        self.numNodes = order + 1 
        data = []
        for i in range(numEle):
            data.append(Element(self.numNodes))

        """
        We need initial condition now
        """
        return data


    def gaussNodes(self, order):
        """
        Return Gauss nodes
        """
        assert(order) > 0
        numNodes = order + 1
        nodes = np.zeros(numNodes)
        if numNodes == 2:
            nodes[0] = -0.577350
            nodes[1] =  0.577350
        elif numNodes == 3:
            nodes[0] = -0.774597
            nodes[1] =  0
            nodes[2] =  0.774597
        elif numNodes == 4:
            nodes[0] = -0.861136
            nodes[1] = -0.339981
            nodes[2] =  0.339981
            nodes[3] =  0.861136
        else:
            raise Exception("Order not supported")
        return nodes

if __name__=="__main__":
    run = Field()
    print(run.gaussNodes(3))

