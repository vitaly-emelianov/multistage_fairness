import numpy as np
import pulp
from itertools import product
from optimizer.solver import Solver


class Optimal(Solver):

    def fit(self, alpha, fairness_type=None, fairness_def=None, num_stage=2):
        lp = pulp.LpProblem("Fair Multistage Selection", pulp.LpMaximize)
        num_feat_per_stage = self.num_feat // num_stage

        # Initializing variables
        phat = {}
        for stage in range(1, num_stage + 1):
            phat[stage] = []
            for i in list(product(range(2), repeat=stage * num_feat_per_stage)):
                phat[stage].append(
                    pulp.LpVariable('p{}-{}'.format(stage, i), lowBound=0, upBound=1, cat='Continuous'))
            phat[stage] = np.array(phat[stage]).reshape([2] * (stage * num_feat_per_stage))

        # Initializing utility function
        lp += (phat[num_stage] * ((self.p * self.py).sum(0) + 1e-10)).sum() / alpha[-1], "precision"

        # Constraints on  selection sizes
        for stage in range(1, num_stage):
            lp += (self.p.sum(0).sum(axis=tuple(range(stage * num_feat_per_stage, self.num_feat))) * phat[
                stage]).sum() <= alpha[
                      stage - 1], "size-{}".format(stage)
        lp += (self.p.sum(0) * phat[num_stage]).sum() == alpha[num_stage - 1], "size_{}".format(num_stage)

        # Constraints on selection probabilities
        for stage in range(2, num_stage + 1):
            for t in list(product(range(2), repeat=self.num_feat)):
                lp += phat[stage][t[:stage * num_feat_per_stage]] <= phat[stage - 1][
                    t[:(stage - 1) * num_feat_per_stage]]

        if fairness_def and fairness_type:
            if fairness_def == "dp":
                ptilde = self.p
            elif fairness_def == "eo":
                ptilde = self.p * self.py

            if fairness_type == "gf":
                lp += (ptilde[0] * phat[stage]).sum() / ptilde[0].sum() == (ptilde[1] * phat[stage]).sum() / ptilde[
                    1].sum()
            elif fairness_type == "lf":
                for stage in range(1, num_stage + 1):
                    lp += (ptilde[0].sum(axis=tuple(range(num_feat_per_stage * stage, self.num_feat))) * phat[
                        stage]).sum() / \
                          ptilde[
                              0].sum() == (
                                  ptilde[1].sum(axis=tuple(range(num_feat_per_stage * stage, self.num_feat))) * phat[
                              stage]).sum() / ptilde[1].sum(), "lf_{}".format(stage)
        lp.solve()
        self.lp = lp
        return pulp.value(lp.objective)

    def minimize_violation(self, alpha, num_stage, fairness_def):
        gf_res = self.fit(alpha, fairness_type="gf", fairness_def=fairness_def, num_stage=num_stage)
        lf_res = self.fit(alpha, fairness_type="lf", fairness_def=fairness_def, num_stage=num_stage)
        lp_min = self.lp.deepcopy()
        # lp_max = self.lp.deepcopy()

        tmin = {}
        for stage in range(1, num_stage):
            tmin[stage] = pulp.LpVariable("tmin{}".format(stage), cat="Continuous")
        lp_min += lp_min.objective >= gf_res - 1e-7, "min_utility"
        for stage in range(1, num_stage):
            lp_min += -tmin[stage]
            lp_min += tmin[stage] >= lp_min.constraints["lf_{}".format(stage)]
            lp_min += tmin[stage] >= -lp_min.constraints["lf_{}".format(stage)]
            del lp_min.constraints["lf_{}".format(stage)]
        lp_min.solve()

        # tmax = {}
        # for stage in range(1, num_stage):
        #     tmax[stage] = pulp.LpVariable("tmax{}".format(stage), cat="Continuous")
        # lp_max += lp_max.objective >= gf_res - 1e-7, "min_utility"
        # for stage in range(1, num_stage):
        #     lp_max += tmax[stage]
        #     lp_max += -tmax[stage] >= lp_max.constraints["lf_{}".format(stage)]
        #     lp_max += -tmax[stage] >= -lp_max.constraints["lf_{}".format(stage)]
        #     del lp_max.constraints["lf_{}".format(stage)]
        # lp_max.solve()
        return gf_res, lf_res, -pulp.value(lp_min.objective)
