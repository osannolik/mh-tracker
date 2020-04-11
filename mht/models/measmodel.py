import numpy as np

class ConstantVelocity(object):

    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.__R = np.diag(2*[sigma**2])

    def dimension(self):
        return 2

    def H(self, x):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]])

    def R(self):
        return self.__R

    def h(self, x):
        return np.dot(self.H(x), x)

    def inv_h(self, z):
        # (A.T @ np.linalg.inv(A @ A.T)) @ z
        return np.array([z[0], z[1], 0.0, 0.0])

    def P0(self):
        return np.diag([self.__R[0,0], self.__R[1,1], 1.0, 1.0])

class CoordinatedTurn(object):

    def __init__(self, sigma):
        self.sigma = sigma
        self.__R = np.diag(2*[sigma**2])

    def dimension(self):
        return 2

    def H(self, x):
        return np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0]])

    def R(self):
        return self.__R

    def h(self, x):
        return np.dot(self.H(x), x)

class Bearing(object):

    def __init__(self, sigma, pos):
        self.sigma = sigma
        self.__R = sigma**2
        self.pos = pos

    def dimension(self):
        return 1

    def H(self, x):
        d2 = np.dot(x[0:2] - self.pos)
        tmp = np.array(len(x)*[0.0])
        tmp[0,0:2] = [-(x[1]-self.pos[1]), x[0]-self.pos[0]] / d2
        return tmp

    def R(self):
        return self.__R

    def h(self, x):
        return np.arctan2(x[1]-self.pos[1], x[0]-self.pos[0])

class DualBearing(object):

    def __init__(self, sigma, pos1, pos2):
        self.sigma = sigma
        self.__R = np.diag(2*[sigma**2])
        self.pos1 = pos1
        self.pos2 = pos2

    def dimension(self):
        return 2

    def H(self, x):
        d1_2 = np.dot(x[0:2] - self.pos1)
        d2_2 = np.dot(x[0:2] - self.pos2)
        tmp = np.array([len(x)*[0.0], len(x)*[0.0]])
        tmp[:,0:2] = np.array([
            [-(x[1]-self.pos1[1]), x[0]-self.pos1[0]] / d1_2,
            [-(x[1]-self.pos2[1]), x[0]-self.pos2[0]] / d2_2])
        return tmp

    def R(self):
        return self.__R

    def h(self, x):
        return np.array([
            [np.arctan2(x[1]-self.pos1[1], x[0]-self.pos1[0])],
            [np.arctan2(x[1]-self.pos2[1], x[0]-self.pos2[0])]])

class RangeBearing(object):

    def __init__(self, sigma_range, sigma_bearing, pos):
        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing
        self.__R = np.diag([sigma_range**2, sigma_bearing**2])
        self.pos = pos

    def dimension(self):
        return 2

    def H(self, x):
        d = lambda v: np.sqrt(np.dot(v[0:2]-self.pos))
        tmp = np.array([len(x)*[0.0], len(x)*[0.0]])
        tmp[:,0:2] = np.array([
            [ (x[0]-self.pos[0]), x[1]-self.pos[1]] / d(x),
            [-(x[1]-self.pos[1]), x[0]-self.pos[0]] / (d(x)**2)])
        return tmp

    def R(self):
        return self.__R

    def h(self, x):
        return np.array([
            [np.sqrt(np.dot(x[0:2]-self.pos))],
            [np.arctan2(x[1]-self.pos[1], x[0]-self.pos[0])]])
