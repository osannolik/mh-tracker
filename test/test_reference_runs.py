import unittest

import numpy as np
import scipy.io
from scipy.stats.distributions import chi2

from mht.tracker import (Tracker)
from mht.gaussian import (Density)
from mht.measmodel import (ConstantVelocity)
from mht.motionmodel import (ConstantVelocity2D)

class TwoTargetsReferenceRun(unittest.TestCase):

    def setUp(self):
        self.f = scipy.io.loadmat('test/measdata_2tgts_man_ref.mat')

        X0 = [
            Density(x=np.array([0.0, 0.0, 1.0, 0.0]), P=np.eye(4)),
            Density(x=np.array([0.0, 10.0, 1.0, 0.0]), P=np.eye(4))
        ]

        self.tracker = Tracker(states=X0)

    def test_1(self):
        measdata = self.f['measdata'].squeeze()
        thmhtest = self.f['TOMHTestimates'].squeeze()

        measmodel = ConstantVelocity(sigma=0.2)
        motionmodel = ConstantVelocity2D(T=1.0, sigma=0.1)

        P_G = 0.99 # size in percentage
        gating_size2 = chi2.ppf(P_G, measmodel.dimension())

        for k, Z in enumerate(measdata):
            detections = list(Z.T)

            self.tracker.update(
                detections,
                P_D=0.90,
                lambda_c=1.0,
                range_c=(-10, 100, -10, 10),
                gating_size2=gating_size2,
                measmodel=measmodel
            )

            est = self.tracker.estimates()

            for trid, state in est.items():
                # assume track-id is naturals 0,1,...
                ref_state_x = thmhtest[k][:,trid].T
                self.assertTrue(np.allclose(state.x, ref_state_x), 
                    'Estimated state for target {} not equal to reference at time {} ({} != {})'.format(
                        trid, k, state.x, ref_state_x
                ))

            self.tracker.predict(motionmodel)

if __name__ == '__main__':
    unittest.main()
