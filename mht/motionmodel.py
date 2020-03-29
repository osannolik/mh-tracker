from numpy import (array, dot, sin, cos, diag, transpose)

DT_DEFAULT=1.0

class ConstantVelocity2D(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def dimension(self):
        return 4
    
    def F(self, x, dt=DT_DEFAULT):
        return array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])

    def Q(self, dt=DT_DEFAULT):
        return array([
            [dt**4/4,      0, dt**3/2,       0],
            [0,      dt**4/4,       0, dt**3/2],
            [dt**3/2,      0,   dt**2,       0],
            [0,      dt**3/2,       0,   dt**2]
        ]) * (self.sigma**2)

    def f(self, x, dt=DT_DEFAULT):
        return dot(self.F(x, dt), x)

class CoordinatedTurn2D(object):

    def __init__(self, sigma_vel, sigma_angle_vel):
        G = array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]])
        S = diag([sigma_vel**2, sigma_angle_vel**2])
        self.__Q = G @ S @ transpose(G)
        #dot(dot(G, S), transpose(G))
        #G.dot(S).dot(transpose(G))

    def dimension(self):
        return 5

    def F(self, x, dt=DT_DEFAULT):
        return array([
            [1, 0, dt*cos(x[3]), -dt*x[2]*sin(x[3]),  0],
            [0, 1, dt*sin(x[3]),  dt*x[2]*cos(x[3]),  0],
            [0, 0,            1,                  0,  0],
            [0, 0,            0,                  1, dt],
            [0, 0,            0,                  0,  1]
        ])

    def Q(self, dt=DT_DEFAULT):
        return self.__Q

    def f(self, x, dt=DT_DEFAULT):
        dx = dt * array([
            x[2]*cos(x[3]),
            x[2]*sin(x[3]),
            0,
            x[4],
            0
        ])
        return x + dx
