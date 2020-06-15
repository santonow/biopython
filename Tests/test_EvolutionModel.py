# Copyright (C) 2020 by Stanislaw Antonowicz (stas.antonowicz@gmail.com)
# This code is part of the Biopython distribution and governed by its
# license. Please see the LICENSE file that should have been included
# as part of this package.

"""Unit tests for the Bio.Phylo.EvolutionModel module."""

import unittest

from Bio.Phylo.EvolutionModel import F81Model
from Bio.Phylo.EvolutionModel import GTRModel


class EvolutionModelTest(unittest.TestCase):
    """Test both F81Model and GTRModel."""

    def test_init_distribution(self):
        for model in [F81Model, GTRModel]:
            with self.subTest(line=model):
                # doesn't sum to one
                with self.assertRaises(ValueError):
                    stat_params = dict(zip("ACGT", [0.5, 0.5, 0.5, 0.5]))
                    evo_model = model(stat_params=stat_params)
                # value less than zero
                with self.assertRaises(ValueError):
                    stat_params = dict(zip("ACGT", [-100, 0.5, 0.5, 0.5]))
                    evo_model = model(stat_params=stat_params)
                # value bigger than one
                with self.assertRaises(ValueError):
                    stat_params = dict(zip("ACGT", [100, 0.5, 0.5, 0.5]))
                    evo_model = model(stat_params=stat_params)

    def test_stat_params_type(self):
        for model in [F81Model, GTRModel]:
            with self.subTest(line=model):
                with self.assertRaises(ValueError):
                    evo_model = model(stat_params="foo")


class F81ModelTest(unittest.TestCase):
    def setUp(self):
        self.evo_model_1 = F81Model()  # uniform
        self.evo_model_2 = F81Model(dict(zip("ACGT", [0.2, 0.3, 0.35, 0.15])))

    def test_valid_probability(self):
        for t in [0, 0.1, 1, 10, 1000, 1000000]:
            with self.subTest(line=t):
                self.assertLessEqual(self.evo_model_1.get_probability("A", "C", t), 1)
                self.assertLessEqual(self.evo_model_2.get_probability("A", "C", t), 1)

    def test_sum_one(self):
        for nuc1 in "ACGT":
            with self.subTest(line=nuc1):
                self.assertAlmostEqual(
                    sum(
                        self.evo_model_1.get_probability(nuc1, nuc2, 1)
                        for nuc2 in "ACGT"
                    ),
                    1,
                )
                self.assertAlmostEqual(
                    sum(
                        self.evo_model_1.get_probability(nuc1, nuc2, 100)
                        for nuc2 in "ACGT"
                    ),
                    1,
                )
                self.assertAlmostEqual(
                    sum(
                        self.evo_model_2.get_probability(nuc1, nuc2, 1)
                        for nuc2 in "ACGT"
                    ),
                    1,
                )
                self.assertAlmostEqual(
                    sum(
                        self.evo_model_2.get_probability(nuc1, nuc2, 100)
                        for nuc2 in "ACGT"
                    ),
                    1,
                )


class GTRModelTest(unittest.TestCase):
    def setUp(self):
        self.evo_model_1 = GTRModel()  # uniform
        self.evo_model_2 = GTRModel(dict(zip("ACGT", [0.2, 0.3, 0.35, 0.15])))  # F81
        self.evo_model_3 = GTRModel(
            dict(zip("ACGT", [0.2, 0.3, 0.35, 0.15])), [1, 2, 3, 4, 5, 6]
        )  # GTR

    def test_valid_probability(self):
        for t in [0, 0.1, 1, 10, 1000, 1000000]:
            with self.subTest(line=t):
                self.assertLessEqual(self.evo_model_1.get_probability("A", "C", t), 1)
                self.assertLessEqual(self.evo_model_2.get_probability("A", "C", t), 1)
                self.assertLessEqual(self.evo_model_3.get_probability("A", "C", t), 1)

    def test_sum_one(self):
        for nuc1 in "ACGT":
            with self.subTest(line=nuc1):
                self.assertAlmostEqual(
                    sum(
                        self.evo_model_1.get_probability(nuc1, nuc2, 1)
                        for nuc2 in "ACGT"
                    ),
                    1,
                )
                self.assertAlmostEqual(
                    sum(
                        self.evo_model_1.get_probability(nuc1, nuc2, 100)
                        for nuc2 in "ACGT"
                    ),
                    1,
                )
                self.assertAlmostEqual(
                    sum(
                        self.evo_model_2.get_probability(nuc1, nuc2, 1)
                        for nuc2 in "ACGT"
                    ),
                    1,
                )
                self.assertAlmostEqual(
                    sum(
                        self.evo_model_2.get_probability(nuc1, nuc2, 100)
                        for nuc2 in "ACGT"
                    ),
                    1,
                )
                self.assertAlmostEqual(
                    sum(
                        self.evo_model_3.get_probability(nuc1, nuc2, 1)
                        for nuc2 in "ACGT"
                    ),
                    1,
                )
                self.assertAlmostEqual(
                    sum(
                        self.evo_model_3.get_probability(nuc1, nuc2, 100)
                        for nuc2 in "ACGT"
                    ),
                    1,
                )
