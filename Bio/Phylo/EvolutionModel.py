# Copyright (C) 2020 by Stanislaw Antonowicz (stas.antonowicz@gmail.com)

"""Classes and methods for evolution models."""

import math
import numpy as np
from itertools import permutations, combinations
from collections.abc import Mapping, Sequence


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
        self._alphabet = sorted(self.stat_params.keys())

    def get_probability(self, site1, site2, t):
        """Return probability of evolving site1 to site2 in time t.

        This should be implemented in a subclass.
        """
        raise NotImplementedError("Method not implemented!")

    @property
    def alphabet(self):
        """Getter method for alphabet."""
        return self._alphabet

    @property
    def stat_params(self):
        """Getter method for stat_params."""
        return self._stat_params

    @stat_params.setter
    def stat_params(self, value):
        """Setter method for stat_params."""
        self._stat_params = self._validate_stat_params(value)
        self._alphabet = sorted(self.stat_params.keys())

    @staticmethod
    def _validate_stat_params(stat_params):
        """Check whether the stat_params dict represents a valid probability distribution (PRIVATE)."""
        if not isinstance(stat_params, Mapping):
            raise ValueError("stat_params must a be a mapping (dictionary)!")
        if not math.isclose(1, sum(stat_params.values())):
            raise ValueError(
                "stat_params must represent a valid probability distribution!"
            )
        if any(x < 0 for x in stat_params.values()):
            raise ValueError(
                "stat_params values have to be grater than zero!"
            )
        if any(x > 1 for x in stat_params.values()):
            raise ValueError(
                "stat_params values have to be less than one!"
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
        """Initialize the parameters, calculate beta."""
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
            It is common to set one of the rates (e.g. G->T) to 1, so the rates are relative to it.
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
    Initialize without any parametes, so it defaults to JC69.
    >>> from Bio.Phylo.EvolutionModel import GTRModel
    >>> evo_model = GTRModel()
    >>> evo_model.get_probability("A", "C", t=1)
    0.1841007154710684

    Change stationary distribution to a non-uniform one - Felsenstein81.
    >>> evo_model.stat_params = dict(zip("ACGT", [0.2, 0.3, 0.3, 0.2]))
    >>> evo_model.get_probability("A", "C", t=1)
    0.22233294822941482

    Now let's set exch_params using a list. It will be now a GTR model.
    First entry corresponds to A->C and C->A rates, second to A->G and G->A etc.
    >>> evo_model.exch_params = [1, 2, 3, 4, 5, 6]
    >>> evo_model.get_probability("A", "C", t=1)
    0.11773674440501203

    """

    def __init__(self, stat_params=None, exch_params=None):
        """Initialize the parameters, perform spectral decomposition.

        :Parameters:
            stat_params: Dict[str, float]
                Parameters of stationary distribution. Default: {sym: 0.25 for sym in "ACGT"}.
            exch_params: Dict[str, float] or Sequence[float]
                Excheangability parameters. Can be either full dictionary (with n(n-1) entries) or a sequence.
                The sequence should have n!/((n-2)!2!) floats, corresponding to combinations(alphabet, 2) position-wise.
        """
        super().__init__(stat_params)
        if not exch_params:
            exch_params = {
                (sym1, sym2): 1 for sym1, sym2 in permutations(self.alphabet, 2)
            }
        self._exch_params = self._validate_exch_params(exch_params)
        self._validate_keys(self.stat_params, self.exch_params)
        self._sym_to_ind = {sym: i for i, sym in enumerate(self.alphabet)}
        self._Q, self._evals, self._evecs, self._evecs_inv = self._compute_spectral()
        self._exp_Q_t_matrices = {}

    def __str__(self):
        """Print model parameters."""
        ret = "GTRModel\n"
        ret += f"Stationary distribution: {self.stat_params}\n"
        ret += "Excheangability parameters:\n"
        ret += str(self.exch_params) + "\n"
        ret += "Rate matrix (Q matrix):\n"
        ret += str(self._Q)
        return ret

    def __repr__(self):
        return f"GTRModel({repr(self.stat_params)}, {repr(self.exch_params)})"

    def get_probability(self, site1, site2, t):
        """Return probability of evolving site1 to site2 in time t.

        If P(t) = exp(Qt) has been computed, return the proper value.
        Otherwise, compute P(t) = V @ exp(lambda * t) @ V**(-1), where
        lambda is the eigenvalues diagonal matrix and V is the Q eigenvectors matrix.
        The eigenvalues and eigenvectors may be complex, but P(t) will not.
        """
        if t not in self._exp_Q_t_matrices:
            self._exp_Q_t_matrices[t] = self._evecs @ np.diag(np.exp(self._evals * t)) @ self._evecs_inv
        return np.abs(self._exp_Q_t_matrices[t][self._sym_to_ind[site1], self._sym_to_ind[site2]])

    def _compute_spectral(self):
        """Compute and return eigenvalues and eigenvectors of the Q matrix (PRIVATE).

        Returns Q rate matrix, its eigenvalues, eigenvector matrix and its inverse.
        """
        self._exp_Q_t_matrices = {}
        Q = np.empty((len(self.alphabet), len(self.alphabet)))
        for sym1, sym2 in permutations(self.alphabet, 2):
            Q[self._sym_to_ind[sym1], self._sym_to_ind[sym2]] = (
                self.exch_params[(sym1, sym2)] * self.stat_params[sym2]
            )
        for sym in self.alphabet:
            Q[self._sym_to_ind[sym], self._sym_to_ind[sym]] = -sum(
                self.exch_params[(sym, other_sym)] * self.stat_params[other_sym]
                for other_sym in self._alphabet
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
        """Getter method for exch_params."""
        return self._exch_params

    @exch_params.setter
    def exch_params(self, value):
        """Setter method for exch_params.

        Redo spectral decomposition every time exch_params is changed.
        Validate that the keys in stat_params and exch_params match.
        """
        self._validate_keys(self.stat_params, value)
        self._exch_params = self._validate_exch_params(value)
        self._Q, self._evals, self._evecs, self._evecs_inv = self._compute_spectral()

    def _validate_exch_params(self, exch_params):
        """Check if exch_params is symmetric or if it has proper length (PRIVATE)."""
        if isinstance(exch_params, Mapping):
            for sym1, sym2 in permutations(self.alphabet, 2):
                if not math.isclose(
                    exch_params[(sym1, sym2)], exch_params[(sym2, sym1)]
                ):
                    raise ValueError(
                        "Wrong exch_params dict, model has to be time reversible."
                    )
            ret = dict(exch_params)
        elif isinstance(exch_params, Sequence):
            if math.factorial(len(self.alphabet)) / (
                math.factorial(len(self.alphabet) - 2) * 2
            ) != len(exch_params):
                raise ValueError("Wrong number of parameters for exch_params!")
            ret = {}
            for val, (sym1, sym2) in zip(
                exch_params, combinations(sorted(self.alphabet), 2)
            ):
                ret[(sym1, sym2)] = val
                ret[(sym2, sym1)] = val
        else:
            raise ValueError(
                "Can't interpret exch_params as a dict or sequence of values!"
            )
        return ret

    def _validate_keys(self, stat_params, exch_params):
        """Check if exch_params and stat_params keys are compatible (PRIVATE)."""
        if isinstance(exch_params, Mapping):
            exch_keys = {x[0] for x in exch_params.keys()}
            stat_keys = set(stat_params.keys())
            exch_minus_stat = exch_keys - stat_keys
            stat_minus_exch = stat_keys - exch_keys
            if exch_minus_stat:
                raise ValueError(
                    f"exch_params has additional key(s) compared to stat_params: {exch_minus_stat}"
                )
            if stat_minus_exch:
                raise ValueError(
                    f"stat_params has additional key(s) compared to exch_params: {stat_minus_exch}"
                )
