import numpy as np

class Advection:

    def initialize(self):
        return

    def create(self):
        from solution import Grid, Element, Field
        from polynomial import Poly

        g = Grid()
        grid = g.getGrid()

        p = Poly()

        numEle = len(grid)

        order = 1
        f = Field()
        data = f.solution(numEle, order)

        for i in range(numEle):
            data[i].nodes = p.gaussNodes(order)

        return

    def run(self):
        return


if __name__=="__main__":
    run = Advection()
    run.create()
