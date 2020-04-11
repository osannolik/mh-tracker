import numpy as np

from mht.utils.generation import measurements

class Volume(object):

    def __init__(self, P_D, clutter_lambda, init_lambda):
        self._pd = P_D
        self._lambda_c = clutter_lambda
        self._lambda_init = init_lambda

    def P_D(self):
        return self._pd

    def _intensity(self, lam):
        return lam/self.volume()

    def clutter_intensity(self, lambda_c=None):
        return self._intensity(self._lambda_c if lambda_c is None else lambda_c)

    def initiation_intensity(self, lambda_init=None):
        return self._intensity(self._lambda_init if lambda_init is None else lambda_init)

    def scan(self, objects, measmodel, ranges):
        assert(ranges.shape[0] == measmodel.dimension())
        objs_inside = {i: x for i, x in objects.items() if self.is_within(measmodel.h(x))}
        return measurements([objs_inside], measmodel, self._pd, self._lambda_c, ranges)[0]

    def volume(self):
        raise NotImplementedError()

    def is_within(self, z):
        raise NotImplementedError()
