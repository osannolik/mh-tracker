from numpy import (array, dot, sin, cos, diag, transpose)

class ConstantVelocity2D(object):

    def __init__(self, T, sigma):
        self.T = T
        self.__Q = sigma**2 * array([
            [T**4/4,     0, T**3/2,     0],
            [0,     T**4/4,     0, T**3/2],
            [T**3/2,     0,   T**2,     0],
            [0,     T**3/2,     0,   T**2]])

    def dimension(self):
        return 4
    
    def F(self, x):
        return array([
            [1, 0, self.T,      0],
            [0, 1,      0, self.T],
            [0, 0,      1,      0],
            [0, 0,      0,      1]])

    def Q(self):
        return self.__Q

    def f(self, x):
        return dot(self.F(x), x)

class CoordinatedTurn2D(object):

    def __init__(self, T, sigma_vel, sigma_angle_vel):
        self.T = T
        G = array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]])
        S = diag([sigma_vel**2, sigma_angle_vel**2])
        self.__Q = G @ S @ transpose(G)
        #dot(dot(G, S), transpose(G))
        #G.dot(S).dot(transpose(G))

    def dimension(self):
        return 5

    def F(self, x):
        return array([
            [1, 0, self.T*cos(x[3]), -self.T*x[2]*sin(x[3]),      0],
            [0, 1, self.T*sin(x[3]),  self.T*x[2]*cos(x[3]),      0],
            [0, 0,                1,                      0,      0],
            [0, 0,                0,                      1, self.T],
            [0, 0,                0,                      0,      1]])

    def Q(self):
        return self.__Q

    def f(self, x):
        dx = array([self.T*x[2]*cos(x[3]),
                    self.T*x[2]*sin(x[3]),
                    0,
                    self.T*x[4],
                    0])
        return x + dx
