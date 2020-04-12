import numpy as np

from mht.constants import (EPS, LARGE, LOG_0, MISS)
from . import data_association

from collections import (OrderedDict)
from copy import (deepcopy)

def _normalize_log_sum(items):
    if len(items) == 0:
        return (items, None)
    elif len(items) == 1:
        log_sum = items[0]
    else:
        i = sorted(range(len(items)), key=lambda k: items[k], reverse=True)
        max_log_w = items[i[0]]
        log_sum = max_log_w + np.log(1.0+sum(np.exp(items[i[1:]]-max_log_w)))

    return (items-log_sum, log_sum)

class LocalHypothesis(object):

    def __init__(self, target, LLR, log_likelihood, LLR_max=None):
        self._target = target
        self._llr = LLR
        self._llr_max = LLR if LLR_max is None else LLR_max
        self._llhood = log_likelihood
        self._lid = self.__class__._counter
        self.__class__._counter += 1

    def id(self):
        return self._lid

    def target(self):
        return deepcopy(self._target)

    def density(self):
        return deepcopy(self._target.density())

    def predict(self, t_now):
        self._target.predict(t_now)

    def log_likelihood_ratio(self):
        return self._llr

    def log_likelihood(self):
        return self._llhood

    def is_dead(self):
        return self._target.is_dead()

    def is_confirmed(self):
        return self._target.is_confirmed()

    @classmethod
    def new_from_hit(cls, self, z, hit_llhood, t_hit):
        target = deepcopy(self._target)
        target.update_hit(z, t_hit)
        llr = self._llr + hit_llhood
        return cls(
            target = target,
            LLR = llr,
            log_likelihood = hit_llhood,
            LLR_max = max(self._llr_max, llr),
        )

    @classmethod
    def new_from_miss(cls, self, miss_llhood, t_now):
        target = deepcopy(self._target)
        target.update_miss(t_now)
        llr = self._llr + miss_llhood
        return cls(
            target = target,
            LLR = self._llr + miss_llhood,
            log_likelihood = miss_llhood,
            LLR_max = max(self._llr_max, llr)
        )

    def __repr__(self):
        return "<loc_hyp {0}: {1}>".format(self.id(), self.density())

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

    def is_within(self, volume):
        # Margin using covariance?
        return np.array([lhyp.target().is_within(volume) for lhyp in self._lhyps.values()]).any()

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

    def dead_local_hyps(self):
        return [lid for lid, lhyp in self._lhyps.items() if lhyp.is_dead()]

    def confirmed_local_hyps(self):
        return [lid for lid, lhyp in self._lhyps.items() if lhyp.is_confirmed()]

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

    def predict(self, t_now):
        for lhyp in self._lhyps.values():
            lhyp.predict(t_now)

    def update(self, Z, volume, t_now):
        new_lhyps = dict()

        for lid, lhyp in self._lhyps.items():
            target = lhyp.target()
            (z_in_gate, in_gate_indices) = target.gating(Z)
            lh = np.array([target.predicted_likelihood(z) for z in z_in_gate])
            lhood = np.full(len(Z), LOG_0)
            lhood[in_gate_indices] = lh + np.log(volume.P_D()+EPS) - np.log(volume.clutter_intensity()+EPS)
            new_lhyps[lid] = OrderedDict([
                (j, LocalHypothesis.new_from_hit(lhyp, Z[j], lhood[j], t_now)
                    if j in in_gate_indices else
                    None)
                for j in range(len(Z))
            ])
            P_G = 1.0
            new_lhyps[lid][MISS] = LocalHypothesis.new_from_miss(lhyp, np.log(1.0 - volume.P_D()*P_G + EPS), t_now)

        return new_lhyps

Track._counter = 0

class Tracker(object):

    def __init__(self, max_nof_hyps, hyp_weight_threshold):
        self._M = max_nof_hyps
        self._weight_threshold = hyp_weight_threshold
        self.tracks = dict()
        self.ghyps = [dict()]
        self.gweights = np.array([np.log(1.0)])

    def create_track_trees(self, detections, volume, targetmodel, t_now):
        intensity_c = volume.clutter_intensity()
        intensity_new = volume.initiation_intensity()
        llr0 = np.log(intensity_new+EPS) - np.log(intensity_c+EPS)

        new_ghyp = dict()
        for z in detections:
            llhood = np.log(volume.P_D()+EPS) - np.log(intensity_c+EPS)
            target = targetmodel.from_one_detection(z, t_now)
            new_lhyp = LocalHypothesis(target, llr0, llhood, t_now)
            new_track = Track(new_lhyp)

            self.tracks[new_track.id()] = new_track

            new_ghyp[new_track.id()] = new_lhyp.id()

        total_init_cost = len(new_ghyp) * -llr0

        return new_ghyp, total_init_cost

    def _unnormalized_weight(self, ghyp):
        return sum([self.tracks[trid].log_likelihood(lid) for trid, lid in ghyp.items()])

    def update_global_hypotheses(self, track_updates, Z, targetmodel, volume, M, weight_threshold, t_now):
        new_weights = list()
        new_ghyps = list()

        if self.tracks:

            for ghyp, weight in zip(self.ghyps, self.gweights):
                cost_matrix = data_association.CostMatrix(ghyp, track_updates)

                if cost_matrix.tracks():
                    nof_best = np.ceil(np.exp(weight) * M)

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
                                # Not part of assignment problem, keep the old
                                new_ghyp[trid] = lid

                        init_ghyp, _ = \
                            self.create_track_trees(Z[unassigned_detections], volume, targetmodel, t_now)

                        new_ghyp.update(init_ghyp)

                        weight_delta = self._unnormalized_weight(new_ghyp)

                        new_weights.append(weight + weight_delta)
                        new_ghyps.append(new_ghyp)
                else:
                    # No track in hyp is included in assignment problem, keep the old
                    new_weights.append(weight)
                    new_ghyps.append(ghyp)

        else:
            init_ghyp, _ = self.create_track_trees(Z, volume, targetmodel, t_now)

            new_weights = [self._unnormalized_weight(init_ghyp)]
            new_ghyps = [init_ghyp]

        assert(len(new_ghyps)==len(new_weights))

        new_weights, new_ghyps = self.prune_dead(np.array(new_weights), np.array(new_ghyps))
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
        dead_lhyps = {trid: track.dead_local_hyps() for trid, track in self.tracks.items()}

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

        return np.array(pruned_weights), np.array(pruned_ghyps)

    @staticmethod
    def hypothesis_prune(weights, hypotheses, threshold):
        keep = weights >= threshold
        return (weights[keep], hypotheses[keep])

    @staticmethod
    def hypothesis_cap(weights, hypotheses, M):
        if len(weights) > M:
            i = np.argsort(weights)
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

    def predict(self, t_now):
        for track in self.tracks.values():
            track.predict(t_now)

    def calculate_weights(self, global_hypotheses):
        weights_updated = [
            sum([self.tracks[trid].log_likelihood(lid) for trid, lid in ghyp.items()])
            for ghyp in global_hypotheses
        ]
        gweights, _ = _normalize_log_sum(np.array(weights_updated))
        return gweights

    def terminate_tracks(self):
        trids_in_ghyps = set([trid for ghyp in self.ghyps for trid in ghyp.keys()])
        unused_tracks = set(self.tracks.keys()) - trids_in_ghyps
        for trid in unused_tracks:
            del self.tracks[trid]

    def update(self, detections, volume, targetmodel, t_now):
        Z = np.array(detections)

        track_updates = OrderedDict([
            (trid, track.update(Z, volume, t_now))
            for trid, track in self.tracks.items() if track.is_within(volume)
        ])

        self.update_global_hypotheses(track_updates, Z, targetmodel, volume, self._M, self._weight_threshold, t_now)

        self.terminate_tracks()

    def process(self, detections, volume, targetmodel, t_now):
        self.predict(t_now)
        self.update(detections, volume, targetmodel, t_now)
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
