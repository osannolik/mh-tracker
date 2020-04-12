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

class CartesianVolume(Volume):

    def __init__(self, ranges, P_D, clutter_lambda, init_lambda):
        """
        ranges is [[x0_min, x0_max], [x1_min, x1_max], ...]
        """
        assert(ranges.shape[1]==2)
        assert((ranges[:,0] <= ranges[:,1]).all())
        self._ranges = np.array(ranges)
        super(CartesianVolume, self).__init__(P_D, clutter_lambda, init_lambda)

    def volume(self):
        return (self._ranges[:,1] - self._ranges[:,0]).prod()

    def is_within(self, z):
        return ((self._ranges[:,0] <= z) & (z <= self._ranges[:,1])).all()

    def scan(self, objects, measmodel):
        return super(CartesianVolume, self).scan(objects, measmodel, self._ranges)