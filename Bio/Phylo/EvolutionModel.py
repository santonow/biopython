# Copyright (C) 2020 by Stanislaw Antonowicz (stas.antonowicz@gmail.com)

"""Classes and methods for evolution models."""

import math
from itertools import permutations


class EvolutionModel:
    """A class representing General Time Reversible (GTR) model.

    All common models (JC69, F81, K80) are a specific cases of GTR.
    If no arguments to __init__ are provided, defaults to JC69 model.

    :Parameters:
        exch_params: Dict[Tuple[str, str], float]
            Exchangeability parameters. Default: {(nuc1, nuc2): 1 for nuc1, nuc2 in permutations("ACGT", 2)}.
            The model has to be time reversible, so exch_params[(nuc1, nuc2)] == exch_params[(nuc2, nuc1)].
        stat_params: Dict[str, float]
            Parameters of stationary distribution. Default: {nuc: 0.25 for nuc in "ACGT"}.
        Q: Dict[Tuple[str, str], float]
            A rate matrix.
    """

    def __init__(self, exch_params=None, stat_params=None):
        """Initialize the parameters and calculate Q rate matrix."""
        if not exch_params:
            exch_params = {(nuc1, nuc2): 1 for nuc1, nuc2 in permutations("ACGT", 2)}
        if not stat_params:
            stat_params = {nuc: 0.25 for nuc in "ACGT"}
        for nuc1, nuc2 in permutations("ACGT", 2):
            if exch_params[(nuc1, nuc2)] != exch_params[(nuc2, nuc1)]:
                raise ValueError(
                    """Wrong exch_params dict, model has to be time reversible."""
                )
        # if not (1 - sum(stat_params.values())) < 0.000001:
        #     raise ValueError("stat_params values must sum to 1!")
        self.exch_params = exch_params
        self.stat_params = stat_params
        self.Q = {}
        for nuc1, nuc2 in permutations("ACGT", 2):
            self.Q[(nuc1, nuc2)] = (
                self.exch_params[(nuc1, nuc2)] * self.stat_params[nuc2]
            )
        for nuc in "ACGT":
            self.Q[(nuc, nuc)] = -sum(
                exch_params[(nuc, other_nuc)] * stat_params[other_nuc]
                for other_nuc in "ACGT"
                if other_nuc != nuc
            )

    def get_probability(self, site1, site2, t):
        """Calculate probability of evolving site2 to site1 in time t.

        The time is measured in expected substitutions per site.
        """
        return math.exp(self.Q[(site1, site2)] * t)
