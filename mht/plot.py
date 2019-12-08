import numpy as np
import matplotlib.pyplot as plt

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
