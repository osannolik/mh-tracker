import numpy as np

def mahalanobis2(x, mu, inv_sigma):
    d = x-mu
    return d.T @ inv_sigma @ d

def moment_matching(log_w, densities):
    w = np.exp(log_w)
    x_weighted = np.dot(w, [d.x for d in densities])
    spread = lambda x, mu: (x-mu)[np.newaxis].T @ (x-mu)[np.newaxis]
    P_weighted = sum([w[i] * (d.P + spread(d.x, x_weighted)) for i,d in enumerate(densities)])
    return Density(x_weighted, P_weighted)

def kalman_predict(density, motion, dt):
    F = motion.F(density.x, dt)
    x = motion.f(density.x, dt)
    P = F @ density.P @ F.T + motion.Q(dt)
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
    in_gate = np.array([mahalanobis2(zi, zbar, inv_S) < size2 for zi in Z])
    return (Z[in_gate,:], in_gate)

class Density(object): 
    __slots__ = ('x', 'P')

    def __init__(self, x, P):
        self.x = np.float64(np.array(x))
        self.P = np.float64(np.array(P))

    def __repr__(self):
        return "<density x={0}>".format(self.x)

    def __eq__(self, other):
        if isinstance(other, Density):
            return np.array_equal(self.x, other.x) and np.array_equal(self.P, other.P)
        return NotImplemented

    def cov_ellipse(self, measure=None, nstd=2):
        if measure is not None:
            H = measure.H(self.x)
            Pz = H @ self.P @ H.T
            z = measure.h(self.x)
        else:
            Pz = self.P[0:2,0:2]
            z = self.x[0:2]
        
        eigvals, vecs = np.linalg.eigh(Pz)
        order = eigvals.argsort()[::-1]
        eigvals, vecs = eigvals[order], vecs[:,order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        r1, r2 = nstd * np.sqrt(eigvals)

        return z, r1, r2, theta

    def ln_mvnpdf(self, x):
        ln_det_sigma = np.log(np.linalg.det(self.P))
        inv_sigma = np.linalg.inv(self.P)
        return -0.5 * (ln_det_sigma + mahalanobis2(np.array(x), self.x, inv_sigma) + len(x)*np.log(2*np.pi))

    def gating(self, Z, measure, size2, inv_S=None, bool_index=False):
        if inv_S is None:
            inv_S = np.linalg.inv(innovation(self, measure))

        zbar = measure.h(self.x)
        is_inside = lambda z: mahalanobis2(z, zbar, inv_S) < size2
        if bool_index:
            in_gate = np.array([is_inside(z) for z in Z])
        else:
            in_gate = np.array([i for i, z in enumerate(Z) if is_inside(z)], dtype=int)

#        if len(in_gate) > 0:
        return (Z[in_gate], in_gate)
#        else:
#            return (np.array([]), np.array([]))

    def predicted_likelihood(self, z, measure, S=None):
        zbar = measure.h(self.x)
        d = Density(x=zbar, P=innovation(self, measure) if S is None else S)
        return d.ln_mvnpdf(z)

    def predict(self, motion, dt):
        predicted = kalman_predict(self, motion, dt)
        self.x, self.P = predicted.x, predicted.P
        return self

    def update(self, z, measure, inv_S=None):
        if inv_S is None:
            inv_S = np.linalg.inv(innovation(self, measure))

        updated = kalman_update(self, np.array(z), inv_S, measure)
        self.x, self.P = updated.x, updated.P
        return self

    def kalman_step(self, z, dt, motion, measure):
        for zi in np.array(z):
            self.predict(motion, dt).update(zi, measure)

        return self

    def sample(self):
        return np.random.multivariate_normal(self.x, self.P)

class Mixture(object):

    def __init__(self, weights, components=[]):
        self.weights = np.array(weights)
        self.components = np.array(components)

    def normalize_log_weights(self):
        if len(self.weights) == 1:
            log_sum_w = self.weights[0]
        else:
            i = sorted(range(len(self.weights)), key=lambda k: self.weights[k], reverse=True)
            max_log_w = self.weights[i[0]]
            log_sum_w = max_log_w + np.log(1.0+sum(np.exp(self.weights[i[1:]]-max_log_w)))

        self.weights -= log_sum_w

        return log_sum_w

    def moment_matching(self):
        self.normalize_log_weights()
        self.components = np.array([moment_matching(self.weights, self.components)])
        self.weights = np.zeros(1)
        return Density(x=self.components[0].x, P=self.components[0].P)