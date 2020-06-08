# Copyright (C) 2020 by Magda Grynkiewicz (magda.markowska@gmail.com)

"""Exemplary usage of Bio.Phylo.MCMC classes."""

from Bio.Phylo.EvolutionModel import GTRModel, F81Model
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio import AlignIO
from Bio.Phylo.MCMC import (
    SamplerMCMC,
    LocalWithoutClockStepper,
    ChangeEvolutionParamStepper,
)

# Read the sequences and align
aln = AlignIO.read("msa.phy", "phylip")

# Print the alignment
print(aln)

# Calculate the distance matrix
calculator = DistanceCalculator("identity")
dm = calculator.get_distance(aln)

# Print the distance Matrix
print(dm)

# Construct the phylogenetic tree using UPGMA algorithm
constructor = DistanceTreeConstructor()
tree = constructor.upgma(dm)
# Print the phylogenetic tree in the terminal

Phylo.draw_ascii(tree)

sampler = SamplerMCMC(
    steps_param={
        LocalWithoutClockStepper(1.0): 0.9,
        ChangeEvolutionParamStepper(GTRModel(), 1.0): 0.1,
    }
)
all_results = sampler.get_results(msa=aln, no_iterations=200)
