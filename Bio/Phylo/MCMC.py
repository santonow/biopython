# Copyright (C) 2020 by Stanislaw Antonowicz (stas.antonowicz@gmail.com)

"""Classes and methods for evolution models."""

import random

import math
import numpy as np
from itertools import permutations, combinations
from collections.abc import Mapping, Sequence


class Stepper:
    """Base class for one simulation step.

    :Parameters:
        size_param: double
            Step size parameter sometimes called tuning parameter.
            Should be positive number.
            Default: 1.
    """

    def __init__(self, size_param=None):
        """Init method for Stepper."""
        if not size_param:
            self._size_param = 1
        else:
            self._size_param = self._validate_size_param(size_param)

    def perform_step(self, tree):
        """Return new_tree after performing step on tree.

        This should be implemented in a subclass.
        """
        raise NotImplementedError("Method not implemented!")

    def _change_path_length(self, path):
        """Change all branch_length for clades in path by factor of size_param (PRIVATE).

        Path is a list of consecutive clades from tree.
        Changes branch lengths in place.
        """
        path_length = len(path)
        for i in range(path_length - 1):
            if path[i + 1].is_parent_of(path[i]):
                path[i].branch_length *= self.size_param
            else:
                path[i + 1].branch_length *= self.size_param

    @property
    def size_param(self):
        """Getter method for size_param."""
        return self._size_param

    @size_param.setter
    def size_param(self, value):
        """Setter method for size_param."""
        self._size_param = self._validate_size_param(value)

    @staticmethod
    def _validate_size_param(size_param):
        """Check whether the size_param is greater than zero (PRIVATE)."""
        if size_param <= 0:
            raise ValueError("size_param must a positive number!")
        return size_param


class LocalWithoutClockStepper(Stepper):
    """A class representing local step without the molecular clock.

    :Parameters:
        size_param: double
            Step size parameter sometimes called tuning parameter.
            Default: 1.

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

    def __init__(self, size_param=None):
        """Initialize the size parameter."""
        super().__init__(size_param)

    def perform_step(self, tree):
        """Return new_tree after performing step on tree.

        1. Randomly select 3 consecutive branches with total length m.
        2. Resize m: m' = m * exp(size_param * (U-0.5)) where U ~ uniform[0,1].
        3. Randomly select one of the two outgoing centre branches.
        4. Regraft the selected branch to uniformly chosen position from 0 to m'.
        """
        leafs = tree.get_terminals()
        if len(leafs) < 3:
            raise ValueError("tree must have at least 3 leafs!")
        else:
            all_clades = leafs + tree.get_nonterminals()
            random.shuffle(all_clades)
            first_clade = all_clades.pop()
            helper_clade = all_clades.pop()
            helper_path = [first_clade] + tree.trace(first_clade, helper_clade)
            while len(helper_path) < 4:
                helper_clade = all_clades.pop()
                helper_path = [first_clade] + tree.trace(first_clade, helper_clade)
            path = helper_path[0, 4]
            self._change_path_length(path)

            return tree
