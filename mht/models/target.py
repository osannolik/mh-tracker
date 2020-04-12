from collections import (deque)

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
