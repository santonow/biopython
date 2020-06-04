# Copyright (C) 2020 by Magda Grynkiewicz (magda.markowska@gmail.com)



import random

from Bio.Phylo.TreeConstruction import LikelihoodScorer
from Bio.Phylo.EvolutionModel import GTRModel, F81Model
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio import AlignIO
from Bio.Phylo.MCMC import SamplerMCMC, LocalWithoutClockStepper, ChangeEvolutionParamStepper

# Read the sequences and align
aln = AlignIO.read('msa.phy', 'phylip')

# Print the alignment
print(aln)

# Calculate the distance matrix
calculator = DistanceCalculator('identity')
dm = calculator.get_distance(aln)

# Print the distance Matrix
print('\nDistance Matrix\n===================')
print(dm)


# Construct the phylogenetic tree using UPGMA algorithm
constructor = DistanceTreeConstructor()
tree = constructor.upgma(dm)
# Print the phylogenetic tree in the terminal
print('\nPhylogenetic Tree\n===================')
Phylo.draw_ascii(tree)

sampler = SamplerMCMC(steps_param={LocalWithoutClockStepper(1.0): 0.9, ChangeEvolutionParamStepper(GTRModel(), 1.0): 0.1})
lista_wszystkiego = sampler.get_results(msa=aln, no_iterations=200)

trees = lista_wszystkiego[0]


stepper = LocalWithoutClockStepper(1.0)
new_tree, hr = stepper.perform_step(tree)
Phylo.draw_ascii(tree)

# Print the phylogenetic tree in the terminal
print('\nPhylogenetic Tree\n===================')
Phylo.draw_ascii(tree)

l = ["a", "b", "c"]
p = [0.1, 0.2, 0.7]

evolution_model = GTRModel(stat_params=dict(zip("ACGT", [0.2, 0.2, 0.3, 0.3])), exch_params=[1,2,3,4,5,6])
stepper = ChangeEvolutionParamStepper(evolution_model=F81Model(), size_param=0.1)
evolution_model = stepper.perform_step()
evolution_model.exch_params
isinstance(evolution_model, F81Model)
scorer = LikelihoodScorer(evolution_model=evolution_model)
scorer.get_score(tree, aln)
stepper = LocalWithoutClockStepper(1.0)
stepper.perform_step(tree=tree)
scorer.get_score(tree, aln)
scorer.evolution_model.stat_params[random.choice([*scorer.evolution_model.stat_params.keys()])] = 0.7
sp = scorer.evolution_model.stat_params
