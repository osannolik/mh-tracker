import numpy as np

from .generation import measurements

class CartesianVolume(object):

    def __init__(self, ranges, P_D, clutter_lambda, init_lambda):
        """
        ranges is [[x0_min, x0_max], [x1_min, x1_max], ...]
        """
        assert ranges.shape[1] == 2
        self._ranges = np.array(ranges)
        self._pd = P_D
        self._lambda_c = clutter_lambda
        self._lambda_init = init_lambda

    def P_D(self):
        return self._pd

    def volume(self):
        return abs(self._ranges[:,1] - self._ranges[:,0]).prod()

    def clutter_intensity(self, lambda_c=None):
        pdf_c = 1.0 / self.volume()
        lam = self._lambda_c if lambda_c is None else lambda_c
        return pdf_c * lam

    def initiation_intensity(self, lambda_init=None):
        pdf_n = 1.0 / self.volume()
        lam = self._lambda_init if lambda_init is None else lambda_init
        return pdf_n * lam

    def is_within(self, z):
        return ((self._ranges[:,0] <= z) & (z <= self._ranges[:,1])).all()

    def scan(self, objects, measmodel):
        assert self._ranges.shape[0] == measmodel.dimension()
        objs_inside = {i: x for i, x in objects.items() if self.is_within(measmodel.h(x))}
        return measurements([objs_inside], measmodel, self._pd, self._lambda_c, self._ranges)[0]
