import numpy as np
from numpy import (array, eye)
from numpy.random import (multivariate_normal)

import matplotlib.pyplot as plt

import scipy.io
from scipy.stats.distributions import chi2

from mht.tracker import (Tracker)
from mht.gaussian import (Density)
from mht.measmodel import (ConstantVelocity)
from mht.motionmodel import (ConstantVelocity2D)

f = scipy.io.loadmat('measdata_ref.mat')
measdata = f['measdata'].squeeze()

X0 = [
    Density(x=array([0.0, 0.0, 1.0, 0.0]), P=eye(4))
]

measmodel = ConstantVelocity(sigma=0.2)
motionmodel = ConstantVelocity2D(T=1.0, sigma=0.1)

P_G = 0.999 # size in percentage
gating_size2 = chi2.ppf(P_G, measmodel.dimension())

tracker = Tracker(states=X0)

x_hat = list()
y_hat = list()

for Z in measdata:
    detections = list(Z.T)
    
    # print(detections)

    tracker.update(
        detections,
        P_D=0.90,
        lambda_c=1.0,
        range_c=(-10, 100, -10, 10),
        gating_size2=gating_size2,
        measmodel=measmodel
    )

    state_est = tracker.estimates()
    #print(state_est)

    pos_hat = state_est[0].x[0:2]
    x_hat.append(pos_hat[0])
    y_hat.append(pos_hat[1])
    #print(pos_hat)

    tracker.predict(motionmodel)

plt.plot(x_hat, y_hat, marker='*')

plt.show()
