import motmetrics as mm
import numpy as np

class MOTMetric(object):

    def __init__(self, ground_truth, tracks, max_d2):
        assert len(ground_truth) == len(tracks)

        self.__has_truth = np.array([bool(d) for d in ground_truth]).any()
        self.__has_hyps = np.array([bool(d) for d in tracks]).any()

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

    def summary(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=mm.metrics.motchallenge_metrics, name='metrics')
        return mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )

    def MOTP(self):
        """
        Multiple Object Tracking Precision
        
        The total position error for all matched object-hypothesis, averaged by the total
        number of matches made. It shows the ability of the tracker to estimate precise 
        object positions, independent of its skill at recognizing object configurations,
        keeping consistent trajectories, etc.

        returns a scalar <= 1.0
        """
        if self.__has_hyps and self.__has_truth:
            mh = mm.metrics.create()
            summary = mh.compute(self.acc, metrics=['motp'], return_dataframe=False)
            motp = summary['motp']
            if np.isnan(motp):
                return 0.0
            else:
                return 1.0 - motp
        elif self.__has_hyps:
            return 0.0
        elif self.__has_truth:
            return 0.0
        else:
            return 1.0

    def MOTA(self):
        """
        Multiple Object Tracking Accuracy
        
        The MOT A accounts for all object configuration errors made by the tracker: 
        false positives, misses and mismatches over all object-hypothesis assignments.

        returns a scalar in (-inf, 1.0]
        """
        if self.__has_hyps and self.__has_truth:
            mh = mm.metrics.create()
            summary = mh.compute(self.acc, metrics=['mota'], return_dataframe=False)
            return summary['mota']
        elif self.__has_hyps:
            return 0.0
        elif self.__has_truth:
            return 0.0
        else:
            return 1.0
