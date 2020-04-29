# Copyright (C) 2020 by Stanislaw Antonowicz (stas.antonowicz@gmail.com)

"""Classes and methods for evolution models."""

import math
import numpy as np
from itertools import permutations, product


class EvolutionModel:
    """Base class for evolution models.

    :Parameters:
        stat_params: Dict[str, float]
            A dictionary representing a stationary distribution.
            Default: {nuc: 0.25 for nuc in "ACGT"}.
    """

    def __init__(self, stat_params=None):
        """Init method for GeneralEvolutionModel."""
        if not stat_params:
            self._stat_params = {sym: 0.25 for sym in "ACGT"}
        else:
            self._stat_params = self._validate_stat_params(stat_params)

    def get_probability(self, site1, site2, t):
        """Return probability of evolving site1 to site2 in time t.

        This should be implemented in a subclass.
        """
        raise NotImplementedError("Method not implemented!")

    @property
    def stat_params(self):
        """Getter method for stat_params."""
        return self._stat_params

    @stat_params.setter
    def stat_params(self, value):
        """Setter method for stat_params."""
        self._stat_params = self._validate_stat_params(value)

    @staticmethod
    def _validate_stat_params(stat_params):
        """Check whether the stat_params dict represents a valid probability distribution (PRIVATE)."""
        if not math.isclose(1, sum(stat_params.values())):
            raise ValueError(
                "stat_params must represent a valid probability distribution!"
            )
        return stat_params


class F81Model(EvolutionModel):
    """A class representing Felsenstein81 model.

    :Parameters:
        stat_params: Dict[str, float]
            A dictionary representing a stationary distribution.
            Default: {nuc: 0.25 for nuc in "ACGT"} (JC69 model).

    Examples
    --------
    >>> from Bio.Phylo.EvolutionModel import F81Model
    >>> evo_model = F81Model()
    >>> evo_model.get_probability("A", "C", t=1)
    0.18410071547106832
    >>> evo_model.stat_params = dict(zip("ACGT", [0.2, 0.3, 0.3, 0.2]))
    >>> evo_model.get_probability("A", "C", t=1)
    0.22233294822941482

    """

    def __init__(self, stat_params=None):
        """Initialize the paramters, calculate beta."""
        super().__init__(stat_params)
        self._beta = 1 / (1 - sum(val ** 2 for val in self.stat_params.values()))

    def get_probability(self, site1, site2, t):
        """Calculate probability of evolving site2 to site1 in time t.

        The time is measured in expected substitutions per site.
        """
        expon = np.exp(-self._beta * t)
        if site1 == site2:
            return expon + self.stat_params[site2] * (1 - expon)
        else:
            return self.stat_params[site2] * (1 - expon)

    @EvolutionModel.stat_params.setter
    def stat_params(self, value):
        """Setter method for stat_params.

        Change _beta param every time the stat_params dict is changed.
        """
        EvolutionModel.stat_params.fset(self, value)
        self._beta = 1 / (1 - sum(val ** 2 for val in self.stat_params.values()))


class GTRModel(EvolutionModel):
    """A class representing General Time Reversible (GTR) model.

    All common models (JC69, F81, K80) are a specific cases of GTR.
    If no arguments to __init__ are provided, defaults to JC69 model.
    With specialized class (e.g. F81Model), the probability computation should generally be faster.
    TODO: Check if it works for arbitrary symbol sets (aminoacids, codons).

    :Parameters:
        exch_params: Dict[Tuple[str, str], float]
            Exchangeability parameters. Represents relative rates of substitution.
            It is common to set one of the rates (e.g. G->T) to one, so the rates are relative to it.
            Has to be symmetric (so the model can be time-reversible).
            This relation should be satisfied:
            for all sym1, sym2 in permutations(symbols, 2):
            (exch_params[(sym1, sym2)] == exch_params[(sym2, sym1)]) == True
            Default: {(sym1, sym2): 1 for sym1, sym2 in permutations("ACGT", 2)}.
        stat_params: Dict[str, float]
            Parameters of stationary distribution. Values must sum to one.
            Default: {sym: 0.25 for sym in "ACGT"}.

    Examples
    --------
    >>> from Bio.Phylo.EvolutionModel import GTRModel
    >>> evo_model = GTRModel()
    >>> evo_model.get_probability("A", "C", t=1)
    0.1841007154710684
    >>> evo_model.stat_params = dict(zip("ACGT", [0.2, 0.3, 0.3, 0.2]))
    >>> evo_model.get_probability("A", "C", t=1)
    0.22233294822941482

    """

    def __init__(self, stat_params=None, exch_params=None):
        """Initialize the parameters, perform spectral decomposition."""
        super().__init__(stat_params)
        self._symbols = sorted(set(self.stat_params.keys()))
        if not exch_params:
            exch_params = {
                (sym1, sym2): 1 for sym1, sym2 in permutations(self._symbols, 2)
            }
        self._exch_params = self._validate_exch_params(exch_params)
        self._validate_keys(self.stat_params, self.exch_params)
        self._sym_to_ind = {sym: i for i, sym in enumerate(self._symbols)}
        self._Q, self._evals, self._evecs, self._evecs_inv = self._compute_spectral()

    def __str__(self):
        """Print model parameters."""
        ret = "GTRModel\n"
        ret += "Stationary distribution: {}\n".format(self.stat_params)
        ret += "Excheangability parameters:\n"
        ret += str(self.exch_params) + "\n"
        ret += "Rate matrix (Q matrix):\n"
        ret += str(self._Q)
        return ret

    def __repr__(self):
        return "GTRModel({}, {})".format(repr(self.stat_params), repr(self.exch_params))

    def get_probability(self, site1, site2, t):
        """Return probability of evolving site1 to site2 in time t.

        Basically (V @ exp(lambda * t) @ V^-1)[site1, site2],
        where V is an eigenvectors matrix and lambda is a diagonal eigenvalues matrix.
        """
        return (
            self._evecs[self._sym_to_ind[site1], :] * np.exp(self._evals * t)
        ) @ self._evecs_inv[:, self._sym_to_ind[site2]]

    def _compute_spectral(self):
        """Compute and return eigenvalues and eigenvectors of the Q matrix (PRIVATE).

        Returns Q rate matrix, its eigenvalues, eigenvector matrix and its inverse.
        """
        Q = np.empty((len(self.stat_params.values()), len(self.stat_params.values())))
        for sym1, sym2 in permutations(self._symbols, 2):
            Q[self._sym_to_ind[sym1], self._sym_to_ind[sym2]] = (
                self.exch_params[(sym1, sym2)] * self.stat_params[sym2]
            )
        for sym in self._symbols:
            Q[self._sym_to_ind[sym], self._sym_to_ind[sym]] = -sum(
                self.exch_params[(sym, other_sym)] * self.stat_params[other_sym]
                for other_sym in self._symbols
                if other_sym != sym
            )
        Q = Q / -np.sum(np.array(list(self.stat_params.values())) * np.diag(Q))
        _eigenvals, _eigenvecs = np.linalg.eig(Q)
        _eigenvecs_inv = np.linalg.inv(_eigenvecs)
        return Q, _eigenvals, _eigenvecs, _eigenvecs_inv

    @EvolutionModel.stat_params.setter
    def stat_params(self, value):
        """Setter method for stat_params.

        Redo spectral decomposition every time stat_params is changed.
        Validate that the keys in stat_params and exch_params match.
        """
        self._validate_keys(value, self.exch_params)
        EvolutionModel.stat_params.fset(self, value)
        self._Q, self._evals, self._evecs, self._evecs_inv = self._compute_spectral()

    @property
    def exch_params(self):
        """exch_params getter method."""
        return self._exch_params

    @exch_params.setter
    def exch_params(self, value):
        """Setter method for exch_params.

        Redo spectral decomposition every time exch_params is changed.
        """
        self._validate_keys(self.stat_params, value)
        self._exch_params = self._validate_exch_params(value)
        self._Q, self._evals, self._evecs, self._evecs_inv = self._compute_spectral()

    def _validate_exch_params(self, exch_params):
        """Check if exch_params is symmetric (PRIVATE)."""
        for sym1, sym2 in permutations(self._symbols, 2):
            if not math.isclose(exch_params[(sym1, sym2)], exch_params[(sym2, sym1)]):
                raise ValueError(
                    "Wrong exch_params dict, model has to be time reversible."
                )
        return exch_params

    def _validate_keys(self, stat_params, exch_params):
        """Check if exch_params and stat_params keys are compatible (PRIVATE)."""
        exch_keys = {x[0] for x in exch_params.keys()}
        stat_keys = set(stat_params.keys())
        exch_minus_stat = exch_keys - stat_keys
        stat_minus_exch = stat_keys - exch_keys
        if exch_minus_stat:
            raise ValueError(
                "exch_params has additional key(s) compared to stat_params: {}".format(
                    exch_minus_stat
                )
            )
        if stat_minus_exch:
            raise ValueError(
                "stat_params has additional key(s) compared to exch_params: {}".format(
                    stat_minus_exch
                )
            )
