from numpy import (arctan2, degrees, pi, log, exp, sqrt, dot, array, newaxis, zeros, float64, int, array_equal)
from numpy.linalg import (inv, eigh, det)
from numpy.random import (multivariate_normal)

def mahalanobis2(x, mu, inv_sigma):
    d = x-mu
    return d.T @ inv_sigma @ d

def moment_matching(log_w, densities):
    w = exp(log_w)
    x_weighted = dot(w, [d.x for d in densities])
    spread = lambda x, mu: (x-mu)[newaxis].T @ (x-mu)[newaxis]
    P_weighted = sum([w[i] * (d.P + spread(d.x, x_weighted)) for i,d in enumerate(densities)])
    return Density(x_weighted, P_weighted)

def kalman_predict(density, motion):
    F = motion.F(density.x)
    x = motion.f(density.x)
    P = F @ density.P @ F.T + motion.Q()
    return Density(x=x, P=P)

def kalman_update(density, z, inv_S, measure):
    H = measure.H(density.x)
    K = density.P @ H.T @ inv_S
    x = density.x + K @ (z-measure.h(density.x))
    P = density.P - (K @ H @ density.P)
    return Density(x=x, P=P)

def predicted_likelihood(density, z, S, measure):
    zbar = measure.h(density.x)
    d = Density(x=zbar, P=S)
    return d.ln_mvnpdf(z)

def innovation(density, measure):
    H = measure.H(density.x)
    S = (H @ density.P @ H.T) + measure.R()
    S = 0.5 * (S + S.T) # Ensure positive definite
    return S

def ellipsoidal_gating(density, Z, inv_S, measure, size2):
    zbar = measure.h(density.x)
    in_gate = array([mahalanobis2(zi, zbar, inv_S) < size2 for zi in Z])
    return (Z[in_gate,:], in_gate)

class Density(object): 
    __slots__ = ('x', 'P')

    def __init__(self, x, P):
        self.x = float64(array(x))
        self.P = float64(array(P))

    def __repr__(self):
        return "<density x={0}>".format(self.x)

    def __eq__(self, other):
        if isinstance(other, Density):
            return array_equal(self.x, other.x) and array_equal(self.P, other.P)
        return NotImplemented

    def cov_ellipse(self, measure=None, nstd=2):
        if measure is not None:
            H = measure.H(self.x)
            Pz = H @ self.P @ H.T
            z = measure.h(self.x)
        else:
            Pz = self.P[:]
            z = self.x[:]
        
        eigvals, vecs = eigh(Pz)
        order = eigvals.argsort()[::-1]
        eigvals, vecs = eigvals[order], vecs[:,order]
        theta = degrees(arctan2(*vecs[:, 0][::-1]))
        r1, r2 = nstd * sqrt(eigvals)

        return z, r1, r2, theta

    def ln_mvnpdf(self, x):
        ln_det_sigma = log(det(self.P))
        inv_sigma = inv(self.P)
        return -0.5 * (ln_det_sigma + mahalanobis2(array(x), self.x, inv_sigma) + len(x)*log(2*pi))

    def gating(self, Z, measure, size2, inv_S=None, bool_index=False):
        if inv_S is None:
            inv_S = inv(innovation(self, measure))

        zbar = measure.h(self.x)
        is_inside = lambda z: mahalanobis2(z, zbar, inv_S) < size2
        if bool_index:
            in_gate = array([is_inside(z) for z in Z])
        else:
            in_gate = array([i for i, z in enumerate(Z) if is_inside(z)], dtype=int)

#        if len(in_gate) > 0:
        return (Z[in_gate], in_gate)
#        else:
#            return (array([]), array([]))

    def predict(self, motion):
        predicted = kalman_predict(self, motion)
        self.x, self.P = predicted.x, predicted.P
        return self

    def update(self, z, measure, inv_S=None):
        if inv_S is None:
            inv_S = inv(innovation(self, measure))

        updated = kalman_update(self, array(z), inv_S, measure)
        self.x, self.P = updated.x, updated.P
        return self

    def kalman_step(self, z, motion, measure):
        for zi in array(z):
            self.predict(motion).update(zi, measure)

        return self

    def sample(self):
        return multivariate_normal(self.x, self.P)

class Mixture(object):

    def __init__(self, weights, components=[]):
        self.weights = array(weights)
        self.components = array(components)

    def normalize_log_weights(self):
        if len(self.weights) == 1:
            log_sum_w = self.weights[0]
        else:
            i = sorted(range(len(self.weights)), key=lambda k: self.weights[k], reverse=True)
            max_log_w = self.weights[i[0]]
            log_sum_w = max_log_w + log(1.0+sum(exp(self.weights[i[1:]]-max_log_w)))

        self.weights -= log_sum_w

        return log_sum_w

    def moment_matching(self):
        self.normalize_log_weights()
        self.components = array([moment_matching(self.weights, self.components)])
        self.weights = zeros(1)
        return Density(x=self.components[0].x, P=self.components[0].P)