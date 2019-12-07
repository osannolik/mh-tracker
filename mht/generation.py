import numpy as np
from numpy.random import (multivariate_normal, uniform, poisson)

def ground_truth(t_length, x_birth, t_birth, t_death, motionmodel):
    """
    returns a list of length t_length with dictionaries containing object states. 
    """

    trajs = [dict() for _ in range(t_length)]

    t_births = np.minimum(np.maximum(t_birth, 0), t_length - 1)
    t_deaths = np.minimum(np.maximum(t_death, 0), t_length - 1)

    for i, state in enumerate(x_birth):
        trajs[t_births[i]][i] = state
        for t in range(t_births[i] + 1, t_deaths[i] + 1):
            trajs[t][i] = multivariate_normal(motionmodel.f(trajs[t-1][i]), motionmodel.Q())

    return trajs

def measurements(ground_truth, measmodel, P_D, lambda_c, range_c):
    """
    range_c is [[x0_min, x0_max], [x1_min, x1_max], ...]
    """
    meas = list(list() for _ in range(len(ground_truth)))

    for t, objects in enumerate(ground_truth):
        meas[t] = [
            multivariate_normal(measmodel.h(state), measmodel.R()) 
            for state in objects.values() if uniform() <= P_D
        ]

        delta_c = range_c[:,1] - range_c[:,0]
        for _ in range(poisson(lambda_c)):
            meas[t].append(range_c[:,0] + delta_c * uniform(size=measmodel.dimension()))

    return meas
