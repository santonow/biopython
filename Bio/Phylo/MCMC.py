# Copyright (C) 2020 by Magda Grynkiewicz (magda.markowska@gmail.com)

"""Classes and methods for evolution models."""

import random
import math
import matplotlib as plt
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.EvolutionModel import F81Model, GTRModel, LikelihoodScorer


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
    >>> from Bio.Phylo.MCMC import LocalWithoutClockStepper
    >>> import copy
    >>> stepper = LocalWithoutClockStepper()
    >>> tree = Phylo.read('ncbi_taxonomy.xml', 'phyloxml')
    >>> new_tree = copy.deepcopy(tree)
    >>> stepper.perform_step(tree)
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
            path = helper_path[0:4]
            self._change_path_length(path)
            return tree


class SamplerMCMC:
    """A class representing MCMC sampling scheme.

    :Parameters:
        steps_param: Dict[Stepper, float]
            A dictionary representing MCMC steps distribution.
            Default: {LocalWithoutMolecularClock(): 1}.

    Examples
    --------
    >>> from Bio.Phylo.MCMC import SamplerMCMC
    >>> sampler = SamplerMCMC()
    """

    def __init__(self, steps_param=None):
        """Init method for Stepper."""
        if not steps_param:
            self._steps_param = {LocalWithoutClockStepper: 1}
        else:
            self._steps_param = self._validate_steps_param(steps_param)
        self.trees = []
        self.no_of_consecutive_tree_appearances = []
        self.likelihoods = []
        self.changed_backbone_nodes = []
        self.changed_branching_nodes = []
        self.scorer = LikelihoodScorer(evolution_model=GTRModel())

    @staticmethod
    def _validate_steps_params(steps_params):
        """Check whether the steps_params dict represents a valid probability distribution (PRIVATE)."""
        if not math.isclose(1, sum(steps_params.values())):
            raise ValueError(
                "steps_params must represent a valid probability distribution!"
            )
        return steps_params

    def get_results(self, msa, no_iterations=1000, burn_in=0, plot=False):
        """Perform MCMC sampling procedure.

        1. Construct initial tree from MultipleSequenceAlignment using UPGMA.
        2. For no_iterations perform single step randomly chosen according to steps distribution.
        3. Check if the step is accepted.
        4. Return list[burn_in: no_interations] of: trees, no_of_consecutive_tree_appearances, likelihoods, changed_backbone_nodes, changed_branching_nodes if tree structure unchanged - empty list.
        5. if plot==True: plot likelihoods.
        """
        # validate input
        if not isinstance(msa, MultipleSeqAlignment):
            raise TypeError("Arg msa must be a MultipleSeqAlignment object.")

        if not no_iterations > burn_in:
            raise ValueError("no_interations must be greater than burn_in")

        # build starting tree from MSA using UPGMA algorithm
        calculator = DistanceCalculator("identity")
        distance_matrix = calculator.get_distance(msa)
        constructor = DistanceTreeConstructor()
        current_tree = constructor.upgma(distance_matrix)

        # calc
        likelihood_current = SamplerMCMC.scorer.get_score(current_tree, msa)
        print("\nPhylogenetic Tree\n===================")
        Phylo.draw_ascii(current_tree)

        return print(likelihood_current)
