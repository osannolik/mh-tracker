from collections import deque

import numpy as np
from numpy import (array, argsort, ceil, exp, eye, ones, log, vstack, hstack, int)
from numpy.linalg import (inv)
from .gaussian import (predicted_likelihood, innovation)
from .gaussian import (Density)

from murty import Murty

EPS = np.finfo('d').eps
LARGE = np.finfo('d').max
LOG_0 = -LARGE #log(EPS) #np.finfo('d').min
MISS = None

def _normalize_log_sum(items):
    if len(items) == 0:
        return (items, None)
    elif len(items) == 1:
        log_sum = items[0]
    else:
        i = sorted(range(len(items)), key=lambda k: items[k], reverse=True)
        max_log_w = items[i[0]]
        log_sum = max_log_w + log(1.0+sum(exp(items[i[1:]]-max_log_w)))

    return (items-log_sum, log_sum)

class LocalHypothesis(object):

    def __init__(self, state, LLR, log_likelihood, LLR_max=None, hit_history=None):
        self._state = Density(x=state.x, P=state.P)
        self._llr = LLR
        self._llr_max = LLR if LLR_max is None else LLR_max
        self._llhood = log_likelihood
        self._hit_history = deque(maxlen=5) if hit_history is None else hit_history
        self._lid = self.__class__._counter
        self.__class__._counter += 1

    def id(self):
        return self._lid

    def density(self):
        return Density(x=self._state.x, P=self._state.P)

    def predict(self, motionmodel):
        self._state.predict(motionmodel)

    def log_likelihood_ratio(self):
        return self._llr

    def log_likelihood(self):
        return self._llhood

    def is_hit(self):
        return self._hit_history[-1]

    def nof_results(self):
        return len(self._hit_history)

    def m_of_n_hits(self, m):
        return self._hit_history.count(True) >= m

    @classmethod
    def new_from_hit(cls, self, z, measmodel, hit_llhood, inv_S):
        hist = self._hit_history.copy()
        hist.append(True)
        llr = self._llr + hit_llhood
        return cls(
            state = self.density().update(z, measmodel, inv_S),
            LLR = llr,
            log_likelihood = hit_llhood,
            LLR_max = max(self._llr_max, llr),
            hit_history = hist
        )

    @classmethod
    def new_from_miss(cls, self, miss_llhood):
        hist = self._hit_history.copy()
        hist.append(False)
        llr = self._llr + miss_llhood
        return cls(
            state = self.density(),
            LLR = self._llr + miss_llhood,
            log_likelihood = miss_llhood,
            LLR_max = max(self._llr_max, llr),
            hit_history = hist
        )

    def __repr__(self):
        return "<loc_hyp {0}: {1}>".format(self.id(), self._state)

LocalHypothesis._counter = 0

class Track(object):

    def __init__(self, local_hypothesis):
        self._lhyps = {local_hypothesis.id(): local_hypothesis}
        self._trid = self.__class__._counter
        self.__class__._counter += 1

    def __repr__(self):
        return "<track {0}: {1}>".format(self.id(), self._lhyps)

    def id(self):
        return self._trid

    def estimate(self, lhyp_id=None):
        if lhyp_id is None:
            return [lhyp.density() for lhyp in self._lhyps.values()]
        else:
            return self._lhyps[lhyp_id].density()

    def log_likelihood_ratio(self, lhyp_id=None):
        if lhyp_id is None:
            return [lhyp.log_likelihood_ratio() for lhyp in self._lhyps.values()]
        else:
            return self._lhyps[lhyp_id].log_likelihood_ratio()

    def log_likelihood(self, lhyp_id):
        return self._lhyps[lhyp_id].log_likelihood()

    def dead_local_hyps(self):
        return [
            lid for lid, lhyp in self._lhyps.items() 
            if lhyp.nof_results() > 2 and not lhyp.m_of_n_hits(m=2)
        ]

    def confirmed_local_hyps(self):
        return [
            lid for lid, lhyp in self._lhyps.items() 
            if lhyp.nof_results() > 2 and lhyp.m_of_n_hits(m=2)
        ]

    def terminate(self, lhyp_ids):
        self._lhyps = {
            lid: lhyp for lid, lhyp in self._lhyps.items()
            if lid not in lhyp_ids
        }

    def select(self, lhyp_ids):
        self._lhyps = {
            lid: lhyp for lid, lhyp in self._lhyps.items()
            if lid in lhyp_ids
        }

    def predict(self, motionmodel):
        for lhyp in self._lhyps.values():
            lhyp.predict(motionmodel)

    def update(self, Z, gating_size2, volume, measmodel):
        new_lhyps = dict()
        lhood = dict()

        for h, lhyp in self._lhyps.items():
            state = lhyp.density()
            S = innovation(state, measmodel)
            inv_S = inv(S)
            (z_in_gate, in_gate_indices) = state.gating(Z, measmodel, gating_size2, inv_S)
            lh = array([predicted_likelihood(state, z, S, measmodel) for z in z_in_gate])
            lhood[h] = np.full(len(Z), LOG_0)
            lhood[h][in_gate_indices] = lh + log(volume.P_D()+EPS) - log(volume.clutter_intensity()+EPS)
            new_lhyps[h] = {
                j: LocalHypothesis.new_from_hit(lhyp, Z[j], measmodel, lhood[h][j], inv_S)
                for j in in_gate_indices
            }
            P_G = 1.0
            new_lhyps[h][MISS] = LocalHypothesis.new_from_miss(lhyp, log(1.0 - volume.P_D()*P_G + EPS))

        for _, det_to_new_lhyp in new_lhyps.items():
            for lhyp in det_to_new_lhyp.values():
                self._lhyps[lhyp.id()] = lhyp

        return lhood, new_lhyps

Track._counter = 0

class CostMatrix(object):

    def __init__(self, global_hypothesis, detection_likelihood, miss_likelihood):
        self._index_to_trid = list(global_hypothesis.keys())

        c_track_detection = vstack(tuple(
            (detection_likelihood[trid][global_hypothesis[trid]] 
            for trid in self._index_to_trid)
        ))
        
        n, _ = c_track_detection.shape

        c_miss = np.full((n,n), LOG_0)
        np.fill_diagonal(c_miss, miss_likelihood)

        self._matrix = -1.0 * hstack((c_track_detection, c_miss))

    def solutions(self, max_nof_solutions):
        murty_solver = Murty(self._matrix)

        for _ in range(int(max_nof_solutions)):
            is_ok, sum_cost, track_to_det = murty_solver.draw()

            if not is_ok:
                return None

            n, m_plus_n = self._matrix.shape

            assignments = {
                self._index_to_trid[track_index]: 
                detection if detection in range(m_plus_n - n) else MISS
                for track_index, detection in enumerate(track_to_det)
            }

            unassigned_detections = [
                detection for detection in range(m_plus_n - n)
                if detection not in track_to_det
            ]

            yield sum_cost, assignments, array(unassigned_detections, dtype=int)

    def __repr__(self):
        return "{0}".format(self._matrix)

class Tracker(object):

    def __init__(self, max_nof_hyps, hyp_weight_threshold, states=[]):
        self._M = max_nof_hyps
        self._weight_threshold = hyp_weight_threshold
        
        self.tracks = dict()

        ghyp = dict()
        for state in states:
            local_hypothesis = LocalHypothesis(Density(state.x, state.P), 0.0, 0.0)
            track = Track(local_hypothesis)
            self.tracks[track.id()] = track
            ghyp[track.id()] = local_hypothesis.id()

        self.ghyps = [ghyp]
        self.gweights = array([log(1.0)])

    def create_track_trees(self, detections, volume, measmodel):
        intensity_c = volume.clutter_intensity()
        intensity_new = volume.initiation_intensity()
        llr0 = log(intensity_new+EPS) - log(intensity_c+EPS)

        new_ghyp = dict()
        for z in detections:
            x0 = measmodel.inv_h(z)
            R = measmodel.R()
            P0 = np.diag([R[0,0], R[1,1], 1.0, 1.0])
            llhood = log(volume.P_D()+EPS) - log(intensity_c+EPS)
            new_lhyp = LocalHypothesis(Density(x0, P0), llr0, llhood)
            new_track = Track(new_lhyp)

            self.tracks[new_track.id()] = new_track

            new_ghyp[new_track.id()] = new_lhyp.id()

        total_init_cost = len(new_ghyp) * -llr0

        return new_ghyp, total_init_cost

    def update_global_hypotheses(self, lhyp_updating, Z, measmodel, volume, M, weight_threshold):
        new_weights = list()
        new_ghyps = list()

        if self.tracks:
            detection_likelihood = {trid: lhood for trid, (lhood,_) in lhyp_updating.items()}
            updated_lhyps = {trid: lhyps for trid, (_,lhyps) in lhyp_updating.items()}

            for ghyp, weight in zip(self.ghyps, self.gweights):
                cost_matrix = CostMatrix(ghyp, detection_likelihood, log(1.0-volume.P_D()-EPS))

                nof_best = ceil(exp(weight) * M)

                for sum_cost, assignment, unassigned_detections in cost_matrix.solutions(nof_best):
                    new_ghyp = dict()
                    for trid, detection in assignment.items():
                        lhyps_from_gates = updated_lhyps[trid][ghyp[trid]]
                        if detection in lhyps_from_gates.keys():
                            new_ghyp[trid] = lhyps_from_gates[detection].id()
                        else:
                            print("Assigned detection {0} outside gate to track {1}: force miss".format(detection, trid))
                            new_ghyp[trid] = lhyps_from_gates[MISS].id()

                    init_ghyp, total_init_cost = \
                        self.create_track_trees(Z[unassigned_detections], volume, measmodel)

                    new_ghyp.update(init_ghyp)

                    new_ghyps.append(new_ghyp)
                    new_weights.append(weight - sum_cost - total_init_cost) # gain = -sum_cost

        else:
            init_ghyp, total_init_cost = self.create_track_trees(Z, volume, measmodel)

            new_ghyps = [init_ghyp]
            new_weights = [-total_init_cost]

        assert(len(new_ghyps)==len(new_weights))

        new_weights, _ = _normalize_log_sum(array(new_weights))

        new_weights, new_ghyps = self.hypothesis_prune(new_weights, array(new_ghyps), weight_threshold)
        new_weights, _ = _normalize_log_sum(new_weights)

        new_weights, new_ghyps = self.hypothesis_cap(new_weights, new_ghyps, M)
        new_weights, _ = _normalize_log_sum(new_weights)

        # Kind of 1-scan MHT pruning...
        for trid, track in self.tracks.items():
            track.select([ghyp[trid] for ghyp in new_ghyps if trid in ghyp.keys()])

        self.gweights = new_weights
        self.ghyps = new_ghyps

    @staticmethod
    def hypothesis_prune(weights, hypotheses, threshold):
        keep = weights >= threshold
        return (weights[keep], hypotheses[keep])

    @staticmethod
    def hypothesis_cap(weights, hypotheses, M):
        if len(weights) > M:
            i = argsort(weights)
            m_largest = i[::-1][:M]
            return (weights[m_largest], hypotheses[m_largest])
        else:
            return (weights, hypotheses)

    def estimates(self, only_confirmed=True):
        if len(self.gweights) > 0:
            index_max = np.argmax(self.gweights)
            return {
                trid: self.tracks[trid].estimate(lid)
                for trid, lid in self.ghyps[index_max].items()
                if not only_confirmed or lid in self.tracks[trid].confirmed_local_hyps()
            }
        else:
            return {}

    def predict(self, motionmodel):
        for track in self.tracks.values():
            track.predict(motionmodel)

    def terminate_tracks(self):
        terminate_lids = {trid: track.dead_local_hyps() for trid, track in self.tracks.items()}

        ghyps_updated = list()
        gweights_updated = list()
        for w, ghyp in zip(self.gweights, self.ghyps):
            ghyp_pruned = {
                trid: lid 
                for trid, lid in ghyp.items() if lid not in terminate_lids[trid]
            }
            llhood_pruned = [
                self.tracks[trid].log_likelihood(lid)
                for trid, lid in ghyp.items() if lid in terminate_lids[trid]
            ]

            if ghyp_pruned:
                ghyps_updated.append(ghyp_pruned)
                gweights_updated.append(w - sum(llhood_pruned))

        for trid, track in self.tracks.items():
            track.terminate(terminate_lids[trid])

        self.gweights, _ = _normalize_log_sum(array(gweights_updated))
        self.ghyps = ghyps_updated

        assert(len(self.ghyps) == len(self.gweights))

        trids_in_ghyps = set([trid for ghyp in self.ghyps for trid in ghyp.keys()])
        unused_tracks = set(self.tracks.keys()) - trids_in_ghyps
        for trid in unused_tracks:
            del self.tracks[trid]

    def update(self, detections, volume, gating_size2, measmodel):
        Z = array(detections)

        lhyp_updating = {
            trid: track.update(Z, gating_size2, volume, measmodel)
            for trid, track in self.tracks.items()
        }

        self.update_global_hypotheses(lhyp_updating, Z, measmodel, volume, self._M, self._weight_threshold)

        self.terminate_tracks()

    def process(self, detections, volume, gating_size2, measmodel, motionmodel):
        self.update(detections, volume, gating_size2, measmodel)
        est = self.estimates()
        self.predict(motionmodel)

        return est

    def debug_print(self, t):
        pass
        #print("[t = {}]".format(t))
        #print(len(self.gweights))
        #print("    Weights =")
        #print(self.gweights)

        #for trid in self.estimates().keys():
        #    print("    Track {} LLR = {}".format(trid, self.tracks[trid].log_likelihood_ratio()))

        #print("")
