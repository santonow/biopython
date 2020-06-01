# Copyright (C) 2020 by Magda Grynkiewicz (magda.markowska@gmail.com)

"""Classes and methods for MCMC sampling procedure."""

import random
import math
import matplotlib as plt
import copy
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import LikelihoodScorer
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.EvolutionModel import GTRModel
from Bio.Phylo.EvolutionModel import EvolutionModel
from Bio.Phylo.EvolutionModel import F81Model


class Stepper:
    """Base class for one simulation step.

    :Parameters:
        size_param: float
            Step size parameter sometimes called tuning parameter.
            Should be positive.
    """

    def __init__(self, size_param=False):
        """Init method for Stepper."""
        if not size_param:
            raise ValueError("!")
        else:
            self._size_param = self._validate_size_param(size_param)

    def perform_step(self, tree):
        """Return new_tree after performing step on tree.

        This should be implemented in a subclass.
        """
        raise NotImplementedError("Method not implemented!")

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
        """Check the size_param, should be positive number (PRIVATE)."""
        if size_param <= 0 or not isinstance(size_param, float):
            raise ValueError("size_param must be a positive float!")
        return size_param


class LocalWithoutClockStepper(Stepper):
    """A class representing local step without the molecular clock.

    :Parameters:
        size_param: float
            Step size parameter sometimes called tuning parameter.
            Lambda parameter which rescales the length of modified branch.
            Should be positive float.

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
        """Initialize the class."""
        super().__init__(size_param)

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

    def perform_step(self, tree):
        """Return new_tree after performing step on tree.

        1. Randomly select 3 consecutive branches with total length m.
        2. Resize m: m' = m * exp(size_param * (U-0.5)) where U ~ uniform[0,1].
        3. Randomly select one of the two outgoing centre branches.
        4. Regraft the selected branch to uniformly chosen position from 0 to m'.
        """
        leafs = tree.get_terminals()
        if len(leafs) < 3:
            raise ValueError("Tree must have at least 3 leafs!")
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


class ChangeEvolutionParamStepper(Stepper):
    """A class representing step which changes one randomly chosen parameter of EvolutionModel.

    :Parameters:
        size_param: float
            Step size parameter sometimes called tuning parameter.
            Standard deviation for random change of one of EvolutionModel params.
            Should be positive float.

    Examples
    --------
    >>> from Bio.Phylo.MCMC import ChangeEvolutionParamStepper
    >>> import copy
    >>> stepper = ChangeEvolutionParamStepper()
    >>> tree = Phylo.read('ncbi_taxonomy.xml', 'phyloxml')
    >>> new_tree = copy.deepcopy(tree)
    >>> stepper.perform_step(tree)
    """

    def __init__(self, evolution_model, size_param=None):
        """Initialize the class."""
        super().__init__(size_param)
        if isinstance(evolution_model, EvolutionModel):
            self.evolution_model = evolution_model
        else:
            raise TypeError("Must provide an EvolutionModel object.")

    def _change_stat_params(stat_params, standard_deviation):
        """Return new stat_params after randomly changing one of them (PRIVATE).

        2. Randomly choose one of the parameters and add random.gauss(0, standard_deviation).
        3. Change randomly chosen second parameter accordingly to retain distribution properties.
        """
        new_stat_params = copy.deepcopy(stat_params)
        symbols = [*new_stat_params.keys()]
        random.shuffle(symbols)
        symbol_to_change = symbols.pop()
        random_update = random.gauss(0, standard_deviation)
        if new_stat_params[symbol_to_change] + random_update <= 0 or new_stat_params[symbol_to_change] + random_update >= 1:
            pass
        else:
            new_stat_params[symbol_to_change] += random_update
            second_symbol = symbols.pop()
            sum_without_second = sum(new_stat_params.values()) - new_stat_params[second_symbol]
            new_stat_params[second_symbol] = 1 - sum_without_second
        return new_stat_params

    def perform_step(self, tree):
        """Return new ModelEvolution object with changed parameters.

        1. Check EvolutionModel subclass, for GTR randomly choose stat_params or exch_params.
        2. Change randomly chosen parameters.
        """
        # we change stat_params for F81 Evolution model - only params for this subclass
        if isinstance(self.evolution_model, F81Model):
            new_stat_params = self._change_stat_params(self.evolution_model.stat_params)
            new_evolution_model = F81Model(stat_params=new_stat_params)

        # for GTR Evolution model we randomly change either stat_params or exch_params
        elif random.random() < 0.5:
            new_stat_params = self._change_stat_params(self.evolution_model.stat_params)
            new_evolution_model = GTRModel(stat_params=new_stat_params, exch_params=self.evolution_model.exch_params)
        else:
            new_exch_params = self._change_exch_params(self.evolution_model.exch_params)
            new_evolution_model = F81Model(stat_params=self.evolution_model.stat_params, exch_params=new_exch_params)

        return new_evolution_model


class SamplerMCMC:
    """A class representing MCMC sampling scheme.

    :Parameters:
        steps_param: Dict[Stepper, float]
            A dictionary representing MCMC steps distribution.
            Default: {LocalWithoutMolecularClock(1): 1}.
            Each key must be an instance of Stepper class, please specify stepper inside parametrs here.

    Examples
    --------
    >>> from Bio.Phylo.MCMC import SamplerMCMC
    >>> sampler = SamplerMCMC()
    """

    def __init__(self, steps_param=None):
        """Init method for Stepper."""
        if not steps_param:
            self._steps_param = {LocalWithoutClockStepper(): 1}
        else:
            self._steps_param = self._validate_steps_param(steps_param)
        self.trees = []
        self.no_of_consecutive_tree_appearances = []
        self.likelihoods = []
        self.changed_backbone_nodes = []
        self.changed_branching_nodes = []

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
        scorer = LikelihoodScorer(evolution_model=GTRModel())
        likelihood_current = scorer.get_score(current_tree, msa)
        print(likelihood_current)

        # initializing Stepper

        for _ in range(no_iterations):

            # one step
            proposal_tree = copy.deepcopy(current_tree)
            stepper = list(self._steps_param.keys())[random.randint(len(self._steps_param))]

        return True
