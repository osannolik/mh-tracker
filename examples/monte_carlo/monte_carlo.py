import os
import sys

example_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(example_path, os.path.pardir))

import numpy as np

from mht.tracker import (Tracker)

# Application implementation
from cv_target import (TargetPosition_CV2D)

from mht.scan_volume import (CartesianVolume)

# Some helpers for testing
from mht.utils import generation
from mht.utils import gaussian
from mht.utils import plot
from mht.utils import metrics

targetmodel = TargetPosition_CV2D
measmodel = targetmodel.measure()

init_mean = np.array([0.0, 0.0, 1.0, 0.0])
init_cov = np.diag([0.0, 10.0, 0.2, 0.0])

init_lambda = 0.05

volume = CartesianVolume(
    ranges = np.array([[-1.0, 100.0], [-20.0, 20.0]]),
    P_D=0.90,
    clutter_lambda = 1.0,
    init_lambda = init_lambda
)

mota = list()
motp = list()

show_plots = False

dt = 1.0

for i in range(500):
    tracker = Tracker(
        max_nof_hyps = 40,
        hyp_weight_threshold = np.log(0.05),
    )

    ground_truth = generation.random_ground_truth(
        t_end = 100,
        init_state_density=gaussian.Density(x=init_mean, P=init_cov),
        init_lambda = init_lambda,
        P_survival = 0.95,
        motionmodel = targetmodel.motion(),
        dt = dt
    )

    measurements = [volume.scan(objs, measmodel) for objs in ground_truth]

    estimations = list()
    for t, detections in enumerate(measurements):
        t_now = dt * t
        estimations.append(tracker.process(detections, volume, targetmodel, t_now))
        #tracker.debug_print(t)

    track_states = [
        {trid: density.x for trid, density in est.items()}
        for est in estimations
    ]

    metric = metrics.MOTMetric(ground_truth, track_states, max_d2=4.0)
    mota.append(metric.MOTA())
    motp.append(metric.MOTP())

    print("t{} MOTA = {}, MOTP = {}".format(i, mota[-1], motp[-1]))

    if show_plots:
        #show_plot = False

        p = plot.Plotter(to_plot_coordinates=measmodel.h)
        p.trajectory_2d(ground_truth, linestyle='-')
        p.measurements_2d(measurements, marker='.', color='k')
        p.trajectory_2d(estimations, linestyle='--')
        p.covariances_2d(estimations, measmodel, edgecolor='k', linewidth=1)

        p.show()

print(">>> Mean metrics: MOTA = {}, MOTP = {}".format(np.array(mota).mean(), np.array(motp).mean()))

import matplotlib.pyplot as plt
plt.plot(mota)
plt.show()