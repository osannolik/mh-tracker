import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

from .gaussian import (Density)

def to_state(obj):
    if isinstance(obj, Density):
        return obj.x
    else:
        return obj

def to_2d(objects, measmodel=None):
    """
    if measmodel is None the objects are assumed to already be measurements
    """
    h = lambda x: x if measmodel is None else measmodel.h(x)

    return [
        {i: h(to_state(x)) for i, x in objs.items()}
        if isinstance(objs, dict) else
        {i: h(to_state(x)) for i, x in enumerate(objs)}
        for objs in objects
    ]

def covariance_ellipse_2d(density, measmodel, nstd=2, **kwargs):
    z, r1, r2, theta = density.cov_ellipse(measmodel, nstd)
    ellip = Ellipse(xy=z, width=2*r1, height=2*r2, angle=theta, **kwargs)
    ellip.set_alpha(0.3)
    plt.gca().add_artist(ellip)

    return ellip

def covariances_2d(objects, measmodel, nstd=2, **kwargs):
    for objs in objects:
        for obj in objs.values():
            covariance_ellipse_2d(obj, measmodel, nstd, **kwargs)

def measurements_2d(detections, **kwargs):
    meas = to_2d(detections)
    trajectory_2d(meas, measmodel=None, linestyle='', **kwargs)

def trajectory_2d(objects, measmodel=None, **kwargs):
    if measmodel is not None:
        assert measmodel.dimension() == 2

    objects_meas = to_2d(objects, measmodel)

    n_objs = max([len(objs) for objs in objects_meas])
    t_length = len(objects_meas)

    zx = np.full((t_length, n_objs), np.nan)
    zy = np.full(zx.shape, np.nan)

    for t, objs in enumerate(objects_meas):
        for i, z in objs.items():
            [zx[t,i], zy[t,i]] = z

    plt.plot(zx, zy, **kwargs)

def show():
    plt.show()
