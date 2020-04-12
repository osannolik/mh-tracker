import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

from .gaussian import (Density)

def to_state(obj):
    if isinstance(obj, Density):
        return obj.x
    else:
        return obj

def show():
    plt.show()

class Plotter(object):

    def __init__(self, to_plot_coordinates, **kwargs):
        self._to_z = to_plot_coordinates
        self._fig = plt.figure(**kwargs)

    def show(self):
        show()

    def _project(self, objects, h):
        return [
            {i: h(to_state(x)) for i, x in objs.items()}
            if isinstance(objs, dict) else
            {i: h(to_state(x)) for i, x in enumerate(objs)}
            for objs in objects
        ]

    def _trajectory(self, objects_meas, **kwargs):
        obj_index_to_id = list(set([ids for objs in objects_meas for ids in objs.keys()]))
        id_to_obj_index = {id: i for i, id in enumerate(obj_index_to_id)}

        n_objs = len(obj_index_to_id)
        t_length = len(objects_meas)

        zx = np.full((t_length, n_objs), np.nan)
        zy = np.full(zx.shape, np.nan)

        for t, objs in enumerate(objects_meas):
            for id, z in objs.items():
                i = id_to_obj_index[id]
                [zx[t,i], zy[t,i]] = z

        self._fig.gca().plot(zx, zy, **kwargs)

    def trajectory_2d(self, objects, measmodel=None, **kwargs):
        h = self._to_z if measmodel is None else measmodel.h
        objects_meas = self._project(objects, h)
        self._trajectory(objects_meas, **kwargs)

    def measurements_2d(self, detections, inv_h=None, **kwargs):
        if inv_h is None:
            objects = self._project(detections, lambda z: z[:2])
        else:
            objects = self._project(self._project(detections, inv_h), self._to_z)

        self._trajectory(objects, linestyle='', **kwargs)

    def covariance_ellipse_2d(self, density, measmodel, nstd=2, **kwargs):
        z, r1, r2, theta = density.cov_ellipse(measmodel, nstd)
        ellip = Ellipse(xy=z, width=2*r1, height=2*r2, angle=theta, **kwargs)
        ellip.set_alpha(0.3)
        self._fig.gca().add_artist(ellip)

        return ellip

    def covariances_2d(self, objects, measmodel=None, nstd=2, **kwargs):
        for objs in objects:
            for obj in objs.values():
                self.covariance_ellipse_2d(obj, measmodel, nstd, **kwargs)
