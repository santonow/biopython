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
    TO_DO
    """

    def __init__(self, size_param=None):
        """Initialize the class."""
        super().__init__(size_param)

    def perform_step(self, tree):
        """Return changed tree and Hastings ratio.

        1. Randomly select 3 consecutive branches with total length m.
        2. Resize m: m' = m * exp(size_param * (U-0.5)) where U ~ uniform[0,1].
        3. Randomly select one of the two outgoing centre branches.
        4. Regraft the selected branch to uniformly chosen position from 0 to m'.
        """
        # get list of 4 consecutive clades

        random_path = self._get_path(tree)
        while random_path == []:
            random_path = self._get_path(tree)

        a = random_path[0]
        u = random_path[1]
        v = random_path[2]
        c = random_path[3]

        em = self._get_path_length(random_path)
        em_prim = em * math.exp(self.size_param * (random.uniform(0, 1) - 0.5))
        hastings_ratio = (em_prim / em) ** 2

        # x = dist(a, u)
        x = self._get_path_length(random_path[0:2])
        # y = dist(a, v)
        y = self._get_path_length(random_path[0:3])

        # choose u or v randomly, unless one is root -> choose the other one
        which_to_regraft = random.choice([u, v])

        # sample regrafting place as random Uniform(0, 1)
        u2 = random.uniform(0, 1)

        # if u is chosen to regraft
        if which_to_regraft == u:
            x_prim = u2 * em_prim
            y_prim = y * em_prim / em
        else:
            x_prim = x * em_prim / em
            y_prim = u2 * em_prim

        # if x_prim < y_prim do not change tree topology, only change branch lengths
        if x_prim < y_prim:
            # new_dist(a, u) = x_prim
            self._set_dist(a, u, x_prim)
            self._set_dist(u, v, y_prim - x_prim)
            self._set_dist(v, c, em_prim - y_prim)
            changed_branching_node = ""
        # change tree topology and branch lengths
        else:
            if a.is_parent_of(u):
                changed_branching_node = self._exchange_children_triple(
                    parents=[a, u, v], children=[u, v, c]
                )
            elif c.is_parent_of(v):
                changed_branching_node = self._exchange_children_triple(
                    parents=[c, v, u], children=[v, u, a]
                )
            elif v.is_parent_of(u):
                changed_branching_node = self._exchange_children_triple(
                    parents=[v, u, v], children=[u, a, c]
                )
            else:
                changed_branching_node = self._exchange_children_triple(
                    parents=[u, v, u], children=[v, c, a]
                )
            self._set_dist(a, v, y_prim)
            self._set_dist(v, u, x_prim - y_prim)
            self._set_dist(u, c, em_prim - x_prim)

        Phylo.draw_ascii(tree)
        return hastings_ratio, random_path, changed_branching_node

    def _get_path(self, tree, no_of_clades=4):
        """Get random path made of (no_of_clades - 1) consecutive branches (PRIVATE).

        Path is represented by a list of no_of_clades consecutive clades from tree.
        """
        leafs = tree.get_terminals()
        if len(leafs) < 3:
            raise ValueError("Tree must have at least 3 leafs!")
        else:
            all_clades = leafs + tree.get_nonterminals()
            random.shuffle(all_clades)
            first_clade = all_clades.pop()
            helper_clade = all_clades.pop()
            helper_path = tree.trace(first_clade, helper_clade)
            if first_clade not in helper_path:
                helper_path = [first_clade] + helper_path
            while len(helper_path) < no_of_clades:
                try:
                    helper_clade = all_clades.pop()
                except IndexError:
                    return []
                helper_path = tree.trace(first_clade, helper_clade)
                if first_clade not in helper_path:
                    helper_path = [first_clade] + helper_path
            return helper_path[0:no_of_clades]

    @staticmethod
    def _get_path_length(path):
        """Get length of all branches on the path (PRIVATE).

        Path is a list of consecutive clades from tree.
        """
        path_size = len(path)
        path_len = 0
        for i in range(path_size - 1):
            if path[i + 1].is_parent_of(path[i]):
                path_len += path[i].branch_length
            else:
                path_len += path[i + 1].branch_length
        return path_len

    @staticmethod
    def _set_dist(clade1, clade2, new_distance):
        """Set distance between two node to a given value (PRIVATE).

        Changes branch length in place.
        """
        if clade2.is_parent_of(clade1):
            clade1.branch_length = new_distance
        else:
            clade2.branch_length = new_distance

    @staticmethod
    def _exchange_children(clade1, clade1_child, clade2, clade2_child):
        """Change tree topology by exchanging chosen children between two parents (PRIVATE).

        Changes tree topology in place.
        """
        # establish children position
        if clade1.clades[0] == clade1_child:
            position1 = 0
        else:
            position1 = 1
        if clade2.clades[0] == clade2_child:
            position2 = 0
        else:
            position2 = 1
        # switch children
        to_switch1 = copy.deepcopy(clade1_child)
        to_switch2 = copy.deepcopy(clade2_child)
        clade1.clades[position1] = to_switch2
        clade2.clades[position2] = to_switch1

    @staticmethod
    def _exchange_children_triple(parents, children):
        """Change tree topology for four connected clades return changed_branching_nodes (PRIVATE).

        Changes tree topology in place.
        Arg parents must be an ordered list of connected clades starting with first clade to prune from.
        Arg children mus be an ordered list of connected clades starting with first clade to prune.
        First parent gets second child with its subtree instead of first child.
        Second parent gets third child with its subtree instead of second child.
        Third parent gets updated second parent (third child)
        Details: Larget, Bret. (2008). Markov chain Monte Carlo algorithms for the Bayesian analysis of phylogenetic trees.
        """
        # exchange indicated parent's child for child from list
        if parents[0].clades[0] == children[0]:
            parents[0].clades[0] = children[1]
        else:
            parents[0].clades[1] = children[1]
        if parents[1].clades[0] == children[1]:
            parents[1].clades[0] = children[2]
            changed_branching_node = parents[1].clades[1]
        else:
            parents[1].clades[1] = children[2]
            changed_branching_node = parents[1].clades[0]
        # exchange indicated parent's child for !parent! (it is also a child but updated in previous step
        if parents[2].clades[0] == children[2]:
            parents[2].clades[0] = parents[1]
        else:
            parents[2].clades[1] = parents[1]
        return changed_branching_node


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
    >>> stepper = ChangeEvolutionParamStepper()
    >>> tree = Phylo.read('ncbi_taxonomy.xml', 'phyloxml')
    >>> new_tree = copy.deepcopy(tree)
    >>> stepper.perform_step(tree)
    TO_DO
    """

    def __init__(self, evolution_model, size_param=None):
        """Initialize the class."""
        super().__init__(size_param)
        if isinstance(evolution_model, EvolutionModel):
            self.evolution_model = evolution_model
        else:
            raise TypeError("Must provide an EvolutionModel object.")

    def perform_step(self):
        """Return new ModelEvolution object with changed parameters.

        1. Check EvolutionModel subclass, for GTR randomly choose stat_params or exch_params.
        2. Change randomly chosen parameters.
        """
        # we change stat_params for F81 Evolution model - only params for this subclass
        if isinstance(self.evolution_model, F81Model):
            new_stat_params = self._change_stat_params(
                self.evolution_model.stat_params, sd=self.size_param
            )
            new_evolution_model = F81Model(stat_params=new_stat_params)

        # for GTR Evolution model we randomly change either stat_params or exch_params
        elif random.random() < 0.5:
            new_stat_params = self._change_stat_params(
                self.evolution_model.stat_params, sd=self.size_param
            )
            new_evolution_model = GTRModel(
                stat_params=new_stat_params,
                exch_params=self.evolution_model.exch_params,
            )
        else:
            new_exch_params = self._change_exch_params(
                self.evolution_model.exch_params,
                alphabet=self.evolution_model.alphabet,
                sd=self.size_param,
            )
            new_evolution_model = GTRModel(
                stat_params=self.evolution_model.stat_params,
                exch_params=new_exch_params,
            )

        print(self.evolution_model.stat_params)
        return new_evolution_model

    @staticmethod
    def _change_stat_params(stat_params, sd):
        """Return new stat_params after randomly changing one of them (PRIVATE).

        1. Randomly choose one of the parameters and add random.gauss(0, sd).
        2. Change randomly chosen second parameter accordingly to retain distribution properties.
        """
        new_stat_params = copy.deepcopy(stat_params)
        symbols = [*new_stat_params.keys()]
        random.shuffle(symbols)
        symbol_to_change = symbols.pop()
        random_update = random.gauss(0, sd)
        if (
            new_stat_params[symbol_to_change] + random_update <= 0
            or new_stat_params[symbol_to_change] + random_update >= 1
        ):
            pass
        else:
            new_stat_params[symbol_to_change] += random_update
            second_symbol = symbols.pop()
            sum_without_second = (
                sum(new_stat_params.values()) - new_stat_params[second_symbol]
            )
            if sum_without_second > 1:
                pass
            else:
                new_stat_params[second_symbol] = 1 - sum_without_second
                return new_stat_params

    @staticmethod
    def _change_exch_params(exch_params, alphabet, sd):
        """Return new stat_params after randomly changing one of them (PRIVATE).

        Randomly choose one of the parameters and multiply by (1 + random.gauss(0, sd)).
        """
        new_exch_params = copy.deepcopy(exch_params)
        symbols = copy.deepcopy(alphabet)
        random.shuffle(symbols)
        symbol1, symbol2 = symbols.pop(), symbols.pop()
        random_update = random.gauss(0, sd)
        if new_exch_params[(symbol1, symbol2)] * (1 + random_update) <= 0:
            pass
        else:
            new_exch_params[(symbol1, symbol2)] *= 1 + random_update
            new_exch_params[(symbol2, symbol1)] *= 1 + random_update
        return new_exch_params


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
    TO_DO
    """

    def __init__(self, steps_param=None, evolution_model=None):
        """Init method for Sampler."""
        if not steps_param:
            self._steps_param = {LocalWithoutClockStepper(1.0): 1.0}
        else:
            self._steps_param = self._validate_steps_params(steps_param)

        if not evolution_model:
            self._evolution_model = GTRModel()
        else:
            self._evolution_model = self._validate_evolution(evolution_model)
        self.trees = []
        self.no_of_consecutive_tree_appearances = []
        self.likelihoods = []
        self.changed_backbone_nodes = []
        self.changed_branching_node = []
        self.parameters = []
        self.no_of_consecutive_parameters_appearances = []

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

        # calculate initial likelihood
        scorer = LikelihoodScorer(self._evolution_model)
        likelihood_current = scorer.get_score(current_tree, msa)
        print(likelihood_current)

        index_tree = 0
        index_param = 0

        self.no_of_consecutive_tree_appearances.append(1)
        self.trees.append(current_tree)
        self.likelihoods.append(likelihood_current)
        self.changed_backbone_nodes.append([])
        self.changed_branching_node.append("")
        if isinstance(self._evolution_model, F81Model):
            self.parameters.append(self._evolution_model.stat_params)
        else:
            self.parameters.append(
                [self._evolution_model.stat_params, self._evolution_model.exch_params]
            )
        self.no_of_consecutive_parameters_appearances.append(1)

        for _ in range(no_iterations):
            # one step
            proposal_tree = copy.deepcopy(current_tree)
            stepper = list(self._steps_param.keys())[
                random.randint(0, len(self._steps_param) - 1)
            ]

            if isinstance(stepper, LocalWithoutClockStepper):
                hastings_ratio, backbone, branching = stepper.perform_step(
                    proposal_tree
                )
                proposal_likelihood = scorer.get_score(
                    tree=proposal_tree, alignment=msa
                )
                acceptance_ratio = (
                    proposal_likelihood - likelihood_current + math.log(hastings_ratio)
                )
                # the step is NOT accepted
                if acceptance_ratio < math.log(random.random()):
                    self.no_of_consecutive_tree_appearances[index_tree] += 1
                else:
                    index_tree += 1
                    current_tree = proposal_tree
                    likelihood_current = proposal_likelihood
                    self.no_of_consecutive_tree_appearances.append(1)
                    self.trees.append(current_tree)
                    self.likelihoods.append(likelihood_current)
                    self.changed_backbone_nodes.append(backbone)
                    self.changed_branching_node.append(branching)
            else:
                self._evolution_model = stepper.perform_step()
                scorer = LikelihoodScorer(evolution_model=self._evolution_model)
                proposal_likelihood = scorer.get_score(current_tree, msa)
                acceptance_ratio = proposal_likelihood - likelihood_current
                # the step is NOT accepted
                if acceptance_ratio < math.log(random.random()):
                    self.no_of_consecutive_parameters_appearances[index_param] += 1
                else:
                    index_param += 1
                    current_tree = proposal_tree
                    likelihood_current = proposal_likelihood
                    if isinstance(self._evolution_model, F81Model):
                        self.parameters.append(self._evolution_model.stat_params)
                    else:
                        self.parameters.append(
                            [
                                self._evolution_model.stat_params,
                                self._evolution_model.exch_params,
                            ]
                        )
                    self.no_of_consecutive_parameters_appearances.append(1)

        return [
            self.trees,
            self.likelihoods,
            self.no_of_consecutive_tree_appearances,
            self.changed_backbone_nodes,
            self.changed_branching_node,
            self.parameters,
            self.no_of_consecutive_parameters_appearances,
        ]

    @staticmethod
    def _validate_steps_params(steps_params):
        """Check whether the steps_params dict represents a valid probability distribution (PRIVATE)."""
        if not math.isclose(1, sum(steps_params.values())):
            raise ValueError(
                "steps_params must represent a valid probability distribution!"
            )
        return steps_params

    @staticmethod
    def _validate_evolution(evolution_model):
        """Check whether the steps_params dict represents a valid probability distribution (PRIVATE)."""
        if not isinstance(evolution_model, EvolutionModel):
            raise ValueError(
                "evolution_model must be an object of class EvolutionModel!"
            )
        return evolution_model
