import numpy as np

from collections import (deque)

from .motionmodel import (ConstantVelocity2D)
from .measmodel import (ConstantVelocity)

from mht.utils.gaussian import (Density)
from scipy.stats.distributions import (chi2)

class Target(object):

    def __init__(self, density, t_now):
        self._density = density
        self._time = t_now
        self._time_hit = t_now
        self._hit_history = deque(maxlen=5)
        self._hit_history.append(True)

    def predict(self, t_now):
        self._density.predict(self.motion(), dt=t_now-self._time)
        self._time = t_now

    def update_hit(self, detection, t_now):
        self._density.update(detection, self.measure())
        self._time_hit = t_now
        self._hit_history.append(True)

    def update_miss(self, t_now):
        self._hit_history.append(False)

    def is_confirmed(self):
        return len(self._hit_history) > 2 and self._hit_history.count(True) >= 2

    def is_dead(self):
        timeout = (self._time-self._time_hit) > self.max_coast_time()
        return timeout or (len(self._hit_history) > 2 and self._hit_history.count(True) < 2)

    def density(self):
        return self._density

    def is_within(self, volume):
        return volume.is_within(z = self.measure().h(self._density.x))

    @classmethod
    def from_one_detection(cls, detection, t_now):
        raise NotImplementedError()

    @classmethod
    def motion(self):
        raise NotImplementedError()

    @classmethod
    def measure(self):
        raise NotImplementedError()

    def gating(self, detections):
        raise NotImplementedError()

    def predicted_likelihood(self, detection):
        raise NotImplementedError()

    def max_coast_time(self):
        raise NotImplementedError()

class TargetCV2D(Target):
    
    _motion = ConstantVelocity2D(sigma=0.01)
    _measure = ConstantVelocity(sigma=0.1)

    def __init__(self, density, t_now):
        super(TargetCV2D, self).__init__(density, t_now)
        P_G = 0.99
        self._gating_size2 = chi2.ppf(P_G, self._measure.dimension())

    @staticmethod
    def _inv_h(z):
        return np.array([z[0], z[1], 0.0, 0.0])

    @classmethod
    def _P0(cls):
        R = cls._measure.R()
        return np.diag([R[0,0], R[1,1], 1.0, 1.0])

    @classmethod
    def from_one_detection(cls, detection, t_now):
        return cls(
            density=Density(x=cls._inv_h(detection), P=cls._P0()),
            t_now=t_now
        )

    @classmethod
    def motion(self):
        return self._motion
    @classmethod
    def measure(self):
        return self._measure

    def gating(self, detections):
        return self._density.gating(detections, self._measure, self._gating_size2)

    def predicted_likelihood(self, detection):
        return self._density.predicted_likelihood(detection, self._measure)

    def max_coast_time(self):
        return 4.0
