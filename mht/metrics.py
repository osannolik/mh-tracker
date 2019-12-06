import motmetrics as mm
import numpy as np

class MOTMetric(object):

    def __init__(self, ground_truth, tracks, max_d2):
        assert len(ground_truth) == len(tracks)

        self.acc = mm.MOTAccumulator(auto_id=True)

        for t, objects in enumerate(ground_truth):
            obj_ids = list(objects.keys())
            obj_states = np.array(list(objects.values()))
            hyp_ids = list(tracks[t].keys())
            hyp_states = np.array(list(tracks[t].values()))

            C = mm.distances.norm2squared_matrix(obj_states, hyp_states, max_d2)

            self.acc.update(
                obj_ids,
                hyp_ids,
                C
            )

    def MOTP(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['motp'], return_dataframe=False)
        return summary['motp']

    def MOTA(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['mota'], return_dataframe=False)
        return summary['mota']
