import unittest

import numpy as np

import mht.generation as gen

from numpy import (array, zeros)
from mht.motionmodel import (ConstantVelocity2D)
from mht.measmodel import (ConstantVelocity)

class GenerationTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_ground_truth(self):
        motionmodel = ConstantVelocity2D(T=1.0, sigma=1.0)
        truth = gen.ground_truth(
            t_length = 10,
            x_birth = [zeros(motionmodel.dimension()), array([0.0, 0.0, 1.0, 0.0])],
            t_birth = [2, 3],
            t_death = [8, 10 + 1],
            motionmodel = motionmodel
        )

        self.assertTrue(True) # TODO

    def test_measurements(self):
        motionmodel = ConstantVelocity2D(T=1.0, sigma=1.0)
        truth = gen.ground_truth(
            t_length = 5,
            x_birth = [zeros(motionmodel.dimension())],
            t_birth = [0],
            t_death = [4],
            motionmodel = motionmodel
        )

        meas = gen.measurements(
            ground_truth = truth,
            measmodel = ConstantVelocity(sigma=1.0),
            P_D = 0.8,
            lambda_c = 1.0,
            range_c = array([
                [-1.0, 1.0],
                [-2.0, 2.0]
            ])
        )

        self.assertTrue(True) # TODO

if __name__ == '__main__':
    unittest.main()
