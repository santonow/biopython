# Copyright (C) 2020 by Stanislaw Antonowicz (stas.antonowicz@gmail.com)

"""Classes and methods for evolution models."""

import math


class EvolutionModel:
    """Base class for evolution models.

    :Parameters:
        stat_params: Dict[str, float]
            A dictionary representing a stationary distribution.
            Default: {nuc: 0.25 for nuc in "ACGT"} (JC69 model).
    """

    def __init__(self, stat_params=None):
        """Init method for GeneralEvolutionModel."""
        if not stat_params:
            self.stat_params = {nuc: 0.25 for nuc in "ACGT"}
        else:
            self.stat_params = stat_params

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
        """Check whether the stat_params dict represents a valid probability distribution."""
        if not math.isclose(1, sum(value.values())):
            raise ValueError("stat_params must represent a valid probability distribution!")
        else:
            self._stat_params = value


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

    def get_probability(self, site1, site2, t):
        """Calculate probability of evolving site2 to site1 in time t.

        The time is measured in expected substitutions per site.
        """
        expon = math.exp(-self._beta * t)
        if site1 == site2:
            return expon + self.stat_params[site2] * (1 - expon)
        else:
            return self.stat_params[site2] * (1 - expon)

    @EvolutionModel.stat_params.setter
    def stat_params(self, value):
        """Change _beta param every time the stat_params dict is changed."""
        print("In F81 setter.")
        EvolutionModel.stat_params.fset(self, value)
        self._beta = 1 / (1 - sum(val ** 2 for val in self.stat_params.values()))
