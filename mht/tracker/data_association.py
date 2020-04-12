import numpy as np

from mht.constants import (LOG_0, MISS)

from murty import Murty

class CostMatrix(object):

    def __init__(self, global_hypothesis, track_updates):
        self._included_trids = [trid for trid in track_updates.keys() if trid in global_hypothesis.keys()]

        if len(self._included_trids)==0:
            self._matrix = np.empty(shape=(0,0))
            return

        new_lhyps = lambda trid: track_updates[trid][global_hypothesis[trid]]

        hit_likelihoods = lambda trid: np.array([
            LOG_0 if lhyp is None else lhyp.log_likelihood()
            for detection, lhyp in new_lhyps(trid).items() if detection is not MISS
        ])

        c_track_detection = np.vstack(tuple(
            (hit_likelihoods(trid) for trid in self._included_trids)
        ))

        miss_likelihood = np.array([
            new_lhyps(trid)[MISS].log_likelihood() for trid in self._included_trids
        ])

        c_miss = np.full(2*(len(miss_likelihood),), LOG_0)
        np.fill_diagonal(c_miss, miss_likelihood)

        self._matrix = -1.0 * np.hstack((c_track_detection, c_miss))

    def tracks(self):
        return self._included_trids[:]

    def solutions(self, max_nof_solutions):
        if not self._matrix.size:
            return None

        murty_solver = Murty(self._matrix)

        # Get back trid and detection nr from matrix indices
        to_trid = lambda t: self._included_trids[t]

        for _ in range(int(max_nof_solutions)):
            is_ok, sum_cost, track_to_det = murty_solver.draw()

            if not is_ok:
                return None

            n, m_plus_n = self._matrix.shape

            assignments = {
                to_trid(track_index): det_index if det_index in range(m_plus_n - n) else MISS
                for track_index, det_index in enumerate(track_to_det)
            }

            unassigned_detections = [
                det_index for det_index in range(m_plus_n - n)
                if det_index not in track_to_det
            ]

            yield sum_cost, assignments, np.array(unassigned_detections, dtype=int)

    def __repr__(self):
        return str(self._matrix)
