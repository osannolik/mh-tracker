from collections import (deque, OrderedDict)

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

    def __init__(self, state, LLR, log_likelihood, t_now, t_hit=None, LLR_max=None, hit_history=None):
        self._state = Density(x=state.x, P=state.P)
        self._time = t_now
        self._time_hit = t_hit
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

    def predict(self, motionmodel, t_now):
        self._state.predict(motionmodel, dt=t_now-self._time)
        self._time = t_now

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

    def is_confirmed(self):
        return self.nof_results() > 2 and self.m_of_n_hits(m=2)

    def is_dead(self, max_coast_time):
        timeout = False if self._time_hit is None else (self._time-self._time_hit) > max_coast_time
        return timeout or (self.nof_results() > 2 and not self.m_of_n_hits(m=2))

    @classmethod
    def new_from_hit(cls, self, z, measmodel, hit_llhood, t_hit, inv_S):
        hist = self._hit_history.copy()
        hist.append(True)
        llr = self._llr + hit_llhood
        return cls(
            state = self.density().update(z, measmodel, inv_S),
            LLR = llr,
            log_likelihood = hit_llhood,
            t_now = self._time,
            t_hit = t_hit,
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
            t_now = self._time,
            t_hit = self._time_hit,
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

    def __call__(self, lhyp_id):
        if lhyp_id in self._lhyps.keys():
            return self._lhyps[lhyp_id]
        else:
            return None

    def add(self, local_hypothesis):
        self._lhyps[local_hypothesis.id()] = local_hypothesis

    def estimate(self, lhyp_id=None):
        if lhyp_id is None:
            return [lhyp.density() for lhyp in self._lhyps.values()]
        else:
            return self._lhyps[lhyp_id].density()

    def is_within(self, volume, measmodel):
        # Margin using covariance?
        zbar = [measmodel.h(density.x) for density in self.estimate()]
        return array([volume.is_within(z) for z in zbar]).any()

    def log_likelihood_ratio(self, lhyp_id=None):
        if lhyp_id is None:
            return [lhyp.log_likelihood_ratio() for lhyp in self._lhyps.values()]
        else:
            return self._lhyps[lhyp_id].log_likelihood_ratio()

    def log_likelihood_ratio_max(self, lhyp_id=None):
        if lhyp_id is None:
            return [lhyp._llr_max for lhyp in self._lhyps.values()]
        else:
            return self._lhyps[lhyp_id]._llr_max

    def log_likelihood(self, lhyp_id):
        return self._lhyps[lhyp_id].log_likelihood()

    def dead_local_hyps(self, max_coast_time):
        return [
            lid for lid, lhyp in self._lhyps.items() if lhyp.is_dead(max_coast_time)
        ]

    def confirmed_local_hyps(self):
        return [
            lid for lid, lhyp in self._lhyps.items() if lhyp.is_confirmed()
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

    def predict(self, motionmodel, t_now):
        for lhyp in self._lhyps.values():
            lhyp.predict(motionmodel, t_now)

    def update(self, Z, gating_size2, volume, measmodel, t_now):
        new_lhyps = dict()

        for lid, lhyp in self._lhyps.items():
            state = lhyp.density()
            S = innovation(state, measmodel)
            inv_S = inv(S)
            (z_in_gate, in_gate_indices) = state.gating(Z, measmodel, gating_size2, inv_S)
            lh = array([predicted_likelihood(state, z, S, measmodel) for z in z_in_gate])
            lhood = np.full(len(Z), LOG_0)
            lhood[in_gate_indices] = lh + log(volume.P_D()+EPS) - log(volume.clutter_intensity()+EPS)
            new_lhyps[lid] = OrderedDict([
                (j, LocalHypothesis.new_from_hit(lhyp, Z[j], measmodel, lhood[j], t_now, inv_S)
                    if j in in_gate_indices else
                    None)
                for j in range(len(Z))
            ])
            P_G = 1.0
            new_lhyps[lid][MISS] = LocalHypothesis.new_from_miss(lhyp, log(1.0 - volume.P_D()*P_G + EPS))

        return new_lhyps

Track._counter = 0

class CostMatrix(object):

    def __init__(self, global_hypothesis, track_updates):
        self._included_trids = [trid for trid in track_updates.keys() if trid in global_hypothesis.keys()]

        new_lhyps = lambda trid: track_updates[trid][global_hypothesis[trid]]

        hit_likelihoods = lambda trid: array([
            LOG_0 if lhyp is None else lhyp.log_likelihood()
            for detection, lhyp in new_lhyps(trid).items() if detection is not MISS
        ])

        c_track_detection = vstack(tuple(
            (hit_likelihoods(trid) for trid in self._included_trids)
        ))

        miss_likelihood = array([
            new_lhyps(trid)[MISS].log_likelihood() for trid in self._included_trids
        ])

        c_miss = np.full(2*(len(miss_likelihood),), LOG_0)
        np.fill_diagonal(c_miss, miss_likelihood)

        self._matrix = -1.0 * hstack((c_track_detection, c_miss))

    def solutions(self, max_nof_solutions):
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

            yield sum_cost, assignments, array(unassigned_detections, dtype=int)

    def __repr__(self):
        return str(self._matrix)

class Tracker(object):

    def __init__(self, max_nof_hyps, hyp_weight_threshold, max_coast_time):
        self._M = max_nof_hyps
        self._weight_threshold = hyp_weight_threshold
        self._max_coast_time = max_coast_time
        self.tracks = dict()
        self.ghyps = [dict()]
        self.gweights = array([log(1.0)])

    def create_track_trees(self, detections, volume, measmodel, t_now):
        intensity_c = volume.clutter_intensity()
        intensity_new = volume.initiation_intensity()
        llr0 = log(intensity_new+EPS) - log(intensity_c+EPS)

        new_ghyp = dict()
        for z in detections:
            x0 = measmodel.inv_h(z)
            R = measmodel.R()
            P0 = np.diag([R[0,0], R[1,1], 1.0, 1.0])
            llhood = log(volume.P_D()+EPS) - log(intensity_c+EPS)
            new_lhyp = LocalHypothesis(Density(x0, P0), llr0, llhood, t_now)
            new_track = Track(new_lhyp)

            self.tracks[new_track.id()] = new_track

            new_ghyp[new_track.id()] = new_lhyp.id()

        total_init_cost = len(new_ghyp) * -llr0

        return new_ghyp, total_init_cost

    def _unnormalized_weight(self, ghyp):
        return sum([
            self.tracks[trid].log_likelihood(lid) for trid, lid in ghyp.items()
        ])

    def update_global_hypotheses(self, track_updates, Z, measmodel, volume, M, weight_threshold, t_now):
        new_weights = list()
        new_ghyps = list()

        if self.tracks:
            if not track_updates:
                return

            for ghyp, weight in zip(self.ghyps, self.gweights):
                cost_matrix = CostMatrix(ghyp, track_updates)

                nof_best = ceil(exp(weight) * M)

                for _, assignment, unassigned_detections in cost_matrix.solutions(nof_best):
                    new_ghyp = dict()
                    for trid, lid in ghyp.items():
                        if trid in assignment.keys():
                            lhyps_from_gates = track_updates[trid][lid]
                            detection = assignment[trid]
                            lhyp = lhyps_from_gates[detection]
                            new_ghyp[trid] = lhyp.id()
                            self.tracks[trid].add(lhyp)
                        else:
                            # not part of assignment problem, keep the old
                            new_ghyp[trid] = lid

                    init_ghyp, _ = \
                        self.create_track_trees(Z[unassigned_detections], volume, measmodel, t_now)

                    new_ghyp.update(init_ghyp)

                    weight_delta = self._unnormalized_weight(new_ghyp)

                    new_weights.append(weight + weight_delta)
                    new_ghyps.append(new_ghyp)

        else:
            init_ghyp, _ = self.create_track_trees(Z, volume, measmodel, t_now)

            new_weights = [self._unnormalized_weight(init_ghyp)]
            new_ghyps = [init_ghyp]

        assert(len(new_ghyps)==len(new_weights))

        new_weights, new_ghyps = self.prune_dead(array(new_weights), array(new_ghyps))
        new_weights, _ = _normalize_log_sum(new_weights)

        assert(len(new_ghyps)==len(new_weights))

        new_weights, new_ghyps = self.hypothesis_prune(new_weights, new_ghyps, weight_threshold)
        new_weights, _ = _normalize_log_sum(new_weights)

        new_weights, new_ghyps = self.hypothesis_cap(new_weights, new_ghyps, M)
        new_weights, _ = _normalize_log_sum(new_weights)

        # Kind of 1-scan MHT pruning...
        for trid, track in self.tracks.items():
            track.select([ghyp[trid] for ghyp in new_ghyps if trid in ghyp.keys()])

        # Remove duplicate global hyps
#        hashable_ghyps = [tuple(d.items()) for d in new_ghyps]
#        unique_index = [hashable_ghyps.index(d) for d in set(hashable_ghyps)]

        self.gweights = new_weights
        self.ghyps = new_ghyps

    def prune_dead(self, weights, global_hypotheses):
        dead_lhyps = {trid: track.dead_local_hyps(self._max_coast_time) for trid, track in self.tracks.items()}

        pruned_weights = list()
        pruned_ghyps = list()
        for weight, ghyp in zip(weights, global_hypotheses):
            w_diff = 0.0
            pruned_ghyp = dict()
            for trid, lid in ghyp.items():
                if lid in dead_lhyps[trid]:
                    w_diff += self.tracks[trid].log_likelihood(lid)
                else:
                    pruned_ghyp[trid] = lid

            if pruned_ghyp:
                pruned_weights.append(weight - w_diff)
                pruned_ghyps.append(pruned_ghyp)

        assert(len(pruned_weights)==len(pruned_ghyps))

        return array(pruned_weights), array(pruned_ghyps)

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

    def predict(self, motionmodel, t_now):
        for track in self.tracks.values():
            track.predict(motionmodel, t_now)

    def calculate_weights(self, global_hypotheses):
        weights_updated = [
            sum([self.tracks[trid].log_likelihood(lid) for trid, lid in ghyp.items()])
            for ghyp in global_hypotheses
        ]
        gweights, _ = _normalize_log_sum(array(weights_updated))
        return gweights

    def terminate_tracks(self):
        trids_in_ghyps = set([trid for ghyp in self.ghyps for trid in ghyp.keys()])
        unused_tracks = set(self.tracks.keys()) - trids_in_ghyps
        for trid in unused_tracks:
            del self.tracks[trid]

    def update(self, detections, volume, gating_size2, measmodel, t_now):
        Z = array(detections)

        track_updates = OrderedDict([
            (trid, track.update(Z, gating_size2, volume, measmodel, t_now))
            for trid, track in self.tracks.items() if track.is_within(volume, measmodel)
        ])

        self.update_global_hypotheses(track_updates, Z, measmodel, volume, self._M, self._weight_threshold, t_now)

        self.terminate_tracks()

    def process(self, detections, volume, gating_size2, measmodel, motionmodel, t_now):
        self.predict(motionmodel, t_now)
        self.update(detections, volume, gating_size2, measmodel, t_now)
        return self.estimates()

    def debug_print(self, t):
        pass
        print("[t = {}]".format(t))
        #print(len(self.gweights))
        #print("Weights =")
        print(self.gweights)
        print(self.ghyps)

        for trid in self.estimates().keys():
            print("    Track {} LLR = {} ({})".format(
               self.tracks[trid], 
               self.tracks[trid].log_likelihood_ratio(), 
               self.tracks[trid].log_likelihood_ratio_max()
            ))

        print("")
