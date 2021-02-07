
import random
import math


class Bullbear_cell:
    def __init__(self):
        #initialisation
        self.momentum_x = 0 
        self.momentum_y = 0

    #graph is: y = 2x^2
    def compute_win_y(self):
        mx = self.momentum_x 
        my = 2* mx * mx 
        self.momentum_y =my
        return my 

    def win(self):
        self.momentum_x += 0.1
        if self.momentum_x > 1:
            self.momentum_x = 1
        change = self.compute_win_y()
        return change

    def lose(self):
        self.momentum_y /= 5
        self.momentum_x = math.sqrt(self.momentum_y/2)

        return self.momentum_y