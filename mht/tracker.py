from numpy import (array, ceil, exp, ones, log, vstack, hstack, int)
from numpy.linalg import (inv)
from .gaussian import (predicted_likelihood, innovation)

class GlobalHypothesis(object):

    def __init__(self, lhyps, weights):
        self._hyps = int(lhyps)
        self._weight = weights #array(n * [log(1.0/n)])

    # def solve_data_association(self, predicted_likelihood, P_D, M):
    #     n = len(self._hyps)
    #     miss_cost = log(1.0-P_D) * ones((n,n))
    #     cost = zeros((n,))
    #     cost = predicted_likelihood[]


    def prune(self, w_min):
        pass

    def cap(self, M):
        pass

    def normalize(self):
        pass

class Track(object):

    def __init__(self, state):
        self._lhyps = array([state])
        self._trid = self.__class__._counter
        self.__class__._counter += 1

    def id(self):
        return self._trid

    def predicted_likelihood(self, Z, gating_size2, P_D, intensity_c, measmodel):
        lhood = log(0.0) * ones((len(self._lhyps),len(Z)))
        #lhood = hstack((lhood, log(1.0-P_D) * ones((n_hyps,len(self._lhyps)))))

        for h, state in enumerate(self._lhyps):
            S = innovation(state, measmodel)
            (z_in_gate, in_gate_indices) = state.gating(Z, measmodel, gating_size2, inv(S))
            lh = array([predicted_likelihood(state, z, S, measmodel) for z in z_in_gate])
            lhood[h, in_gate_indices] = lh + log(P_D) - log(intensity_c)

        return lhood

Track._counter = 0

class Tracker(object):

    def __init__(self, states, gating_size2):
        self.tracks = array([Track(x) for x in states[:]])
        
        #self.ghyps = array([array(len(self.tracks) * [0], dtype=int)])
        n = len(self.tracks)
        self.ghyps = [{track.id(): 0 for track in self.tracks}]
        self.gweights = array(n * [log(1.0/n)])

        # n = len(self.tracks)
        # self.ghyps = array([
        #     GlobalHypothesis(lhyps=array(n * [0]), weights=array(n * [log(1.0/n)]))
        # ])

        self.gsize2 = gating_size2

    def solve_data_association(self, predicted_likelihood, P_D, M):
        new_ghyps = list()
        
        nof_best = ceil(exp(self.gweights) * M)

        for g, ghyp in enumerate(self.ghyps):
            cost = -1.0 * hstack((
                vstack((predicted_likelihood[trid][hypnr, :] for trid, hypnr in ghyp.items())),
                log(1.0-P_D) * ones((len(ghyp),len(ghyp)))
            ))




        # return [ghyp_new: (weight, {track: j})]

    def update(self, detections, P_D, lambda_c, range_c, measmodel):
        Z = array(detections)
        
        # For now, assume 2D volume...
        scan_volume = (range_c[1]-range_c[0]) * (range_c[3]-range_c[0])
        pdf_c = 1.0 / scan_volume
        intensity_c = pdf_c * lambda_c

        lhood = {track.id(): track.predicted_likelihood(Z, self.gsize2, P_D, intensity_c, measmodel) 
                 for track in self.tracks}


        pass
