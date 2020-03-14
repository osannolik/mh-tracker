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

def random_ground_truth(t_length, init_state_density, init_lambda, P_survival, motionmodel):
    """
    Generate a set of ground truth objects. 
    Object birth is modelled as a Poisson process with init_lambda as the expected number of
    births per time step.
    The objects initial state is sampled from init_state_density and its trajectory follows
    motionmodel as long as it is still alive.
    Object death is modelled using a constant probability of survival P_survival.

    returns a list of length t_length with dictionaries containing object states. 
    """
    x_birth = list()
    t_birth = list()
    t_death = list()

    for t in range(t_length):
        for _ in range(poisson(init_lambda)):
            x_birth.append(init_state_density.sample())
            t_birth.append(t)

            t_d = t + 1
            while (uniform() <= P_survival) and (t_d < t_length):
                t_d += 1

            t_death.append(t_d)

    assert(len(x_birth)==len(t_birth)==len(t_death))
    assert((np.array(t_birth) < np.array(t_death)).all())

    return ground_truth(t_length, x_birth, t_birth, t_death, motionmodel)
