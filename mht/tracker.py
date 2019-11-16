import numpy as np
from numpy import (array, ceil, exp, eye, ones, log, vstack, hstack, int)
from numpy.linalg import (inv)
from .gaussian import (predicted_likelihood, innovation)

from murty import Murty

EPS = np.finfo('d').eps
LOG_0 = log(EPS) #np.finfo('d').min


def _normalize_log_sum(items):
    if len(items) == 1:
        log_sum = items[0]
    else:
        i = sorted(range(len(items)), key=lambda k: items[k], reverse=True)
        max_log_w = items[i[0]]
        log_sum = max_log_w + log(1.0+sum(exp(items[i[1:]]-max_log_w)))

    return (log_sum, items-log_sum)

class Track(object):

    def __init__(self, state):
        self._lhyps = array([state])
        self._trid = self.__class__._counter
        self.__class__._counter += 1

    def id(self):
        return self._trid

    def detection_likelihood(self, Z, gating_size2, P_D, intensity_c, measmodel):
        lhood = np.full((len(self._lhyps),len(Z)), LOG_0)

        for h, state in enumerate(self._lhyps):
            S = innovation(state, measmodel)
            (z_in_gate, in_gate_indices) = state.gating(Z, measmodel, gating_size2, inv(S))
            lh = array([predicted_likelihood(state, z, S, measmodel) for z in z_in_gate])
            lhood[h, in_gate_indices] = lh + log(P_D+EPS) - log(intensity_c+EPS)

        return lhood

Track._counter = 0

class CostMatrix(object):

    def __init__(self, global_hypothesis, detection_likelihood, miss_likelihood):
        self._index_to_trid = list(global_hypothesis.keys())

        c_track_detection = vstack(tuple(
            (detection_likelihood[trid][global_hypothesis[trid],:] 
            for trid in self._index_to_trid)
        ))
        
        _, m = c_track_detection.shape

        c_miss = np.full((m,m), LOG_0)
        np.fill_diagonal(c_miss, miss_likelihood)

        self._matrix = -1.0 * hstack((c_track_detection.T, c_miss))

    def solutions(self, max_nof_solutions):
        murty_solver = Murty(self._matrix)

        for _ in range(int(max_nof_solutions)):
            is_ok, sum_cost, det_to_track = murty_solver.draw()

            if not is_ok:
                return None

            assignments = {
                self._index_to_trid[track_index]: detection
                for detection, track_index in enumerate(det_to_track)
                if track_index in range(len(self._index_to_trid))
            }

            yield sum_cost, assignments

    def __repr__(self):
        return "{0}".format(self._matrix)

class Tracker(object):

    def __init__(self, states, gating_size2):
        self.tracks = array([Track(x) for x in states[:]])
        
        self.ghyps = [{track.id(): 0 for track in self.tracks}]
        self.gweights = array([log(1.0)])
        self.gsize2 = gating_size2

    @staticmethod
    def _new_global_hypothesis():
        pass


    def update_global_hypotheses(self, detection_likelihood, P_D, M):
        

        new_weights = list()
        new_ghyps = list()

        for ghyp, weight in zip(self.ghyps, self.gweights):
            print("--- Global hyp {0}".format(ghyp))
            cost_matrix = CostMatrix(ghyp, detection_likelihood, log(1.0-P_D-EPS))

            print(cost_matrix)

            nof_best = ceil(exp(weight) * M)

            for sum_cost, assignment in cost_matrix.solutions(nof_best):
                print((sum_cost, assignment))
                
                gain = -sum_cost
                new_weights.append(weight + gain)

                for trid, detection in assignment.items():
                    lhypnr = ghyp[trid]
                    #z = ...
                    



            pass


        # return [ghyp_new: (weight, {track: j})]

    def update(self, detections, P_D, lambda_c, range_c, measmodel):
        Z = array(detections)
        
        # For now, assume 2D volume...
        scan_volume = (range_c[1]-range_c[0]) * (range_c[3]-range_c[0])
        pdf_c = 1.0 / scan_volume
        intensity_c = pdf_c * lambda_c

        lhood = {track.id(): track.detection_likelihood(Z, self.gsize2, P_D, intensity_c, measmodel) 
                 for track in self.tracks}

        self.update_global_hypotheses(detection_likelihood=lhood, P_D=P_D, M=10)

        pass
