####
#
# The MIT License (MIT)
#
# Copyright 2017, 2018 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

import unittest

import numpy as np
import itertools as it

from collections import OrderedDict

# Import function to test
from retention_cls import retention_cls
from rank_svm_cls import get_pairs_from_order_graph, get_pairs_single_system, get_pairs_multiple_systems
from rank_svm_cls import tanimoto_kernel, tanimoto_kernel_mat, minmax_kernel_mat, minmax_kernel

class Test_upper_lower_set_node (unittest.TestCase):
    def test_simplecases (self):
        cretention = retention_cls()

        # ----------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2,  ("M3", "A"): 3, ("M4", "A"): 4,
                    ("M1", "B"): 1, ("M5", "B"): 2}

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_node (cretention.dG)
        dm, dc = cretention.dmolecules, cretention.dcollections

        self.assertEqual (len (d_moleculecut), len (d_target))
        # Check distances
        self.assertEqual (d_moleculecut[(dm["M1"], dc["A"])][0], {(dm["M1"], dc["B"]): 0})
        self.assertEqual (d_moleculecut[(dm["M1"], dc["B"])][0], {(dm["M1"], dc["A"]): 0})
        self.assertEqual (d_moleculecut[(dm["M1"], dc["A"])][1],
                          {(dm["M1"], dc["B"]): 0, (dm["M2"], dc["A"]): 1,
                           (dm["M3"], dc["A"]): 2, (dm["M4"], dc["A"]): 3,
                           (dm["M5"], dc["B"]): 1})
        self.assertEqual (d_moleculecut[(dm["M2"], dc["A"])][1],
                          {(dm["M3"], dc["A"]): 1, (dm["M4"], dc["A"]): 2})
        self.assertEqual (d_moleculecut[(dm["M2"], dc["A"])][0],
                          {(dm["M1"], dc["A"]): 1, (dm["M1"], dc["B"]): 1})

class Test_get_pairs_from_order_graph (unittest.TestCase):
    def test_simplecases (self):
        cretention = retention_cls()

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M1", "B"), 1), (("M5", "B"), 2)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        for allow_overlap in [True, False]:
            pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = allow_overlap,
                                                d_lower = 0, d_upper = np.inf)
            self.assertEqual (len (pairs), 11)
            self.assertIn ((0, 1), pairs)
            self.assertIn ((0, 2), pairs)
            self.assertIn ((0, 3), pairs)
            self.assertIn ((0, 5), pairs)
            self.assertIn ((1, 2), pairs)
            self.assertIn ((1, 3), pairs)
            self.assertIn ((2, 3), pairs)
            self.assertIn ((4, 1), pairs)
            self.assertIn ((4, 2), pairs)
            self.assertIn ((4, 3), pairs)
            self.assertIn ((4, 5), pairs)

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2),
                                 (("M7", "B"), 3)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        for allow_overlap in [True, False]:
            pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = allow_overlap,
                                                d_lower = 0, d_upper= np.inf)
            self.assertEqual (len (pairs), 18)
            self.assertIn ((0, 1), pairs)
            self.assertIn ((0, 2), pairs)
            self.assertIn ((0, 3), pairs)
            self.assertIn ((0, 4), pairs)
            self.assertIn ((0, 5), pairs)
            self.assertIn ((0, 6), pairs)
            self.assertIn ((1, 2), pairs)
            self.assertIn ((1, 3), pairs)
            self.assertIn ((1, 5), pairs)
            self.assertIn ((1, 6), pairs)
            self.assertIn ((2, 3), pairs)
            self.assertIn ((2, 6), pairs)
            self.assertIn ((4, 2), pairs)
            self.assertIn ((4, 3), pairs)
            self.assertIn ((4, 5), pairs)
            self.assertIn ((4, 6), pairs)
            self.assertIn ((5, 3), pairs)
            self.assertIn ((5, 6), pairs)

    def test_bordercases (self):
        cretention = retention_cls()

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M1", "B"), 2), (("M1", "C"), 3)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        for allow_overlap in [True, False]:
            pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = allow_overlap,
                                                d_lower = 0, d_upper = np.inf)
            self.assertEqual (len (pairs), 0)

        # ----------------------------------------------
        d_target = OrderedDict ([(("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M3", "B"), 2), (("M2", "B"), 3)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = True,
                                            d_lower = 0, d_upper = np.inf)
        self.assertEqual (len (pairs), 8)

        pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = False,
                                            d_lower = 0, d_upper = np.inf)
        self.assertEqual (len (pairs), 0)

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = True, d_lower = 0, d_upper = 0)
        self.assertEqual (len (pairs), 0)

        pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = False, d_lower = 0, d_upper = 0)
        self.assertEqual (len (pairs), 0)

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = True, d_lower = np.inf, d_upper = np.inf)
        self.assertEqual (len (pairs), 0)

        pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = False, d_lower = np.inf, d_upper = np.inf)
        self.assertEqual (len (pairs), 0)

    def test_overlap (self):
        cretention = retention_cls()

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = True, d_lower = 0, d_upper = np.inf)

        self.assertEqual (len (pairs), 17)
        self.assertIn ((0, 1), pairs)
        self.assertIn ((0, 2), pairs)
        self.assertIn ((0, 3), pairs)
        self.assertIn ((0, 5), pairs)
        self.assertIn ((0, 4), pairs)
        self.assertIn ((1, 2), pairs)
        self.assertIn ((1, 3), pairs)
        self.assertIn ((1, 5), pairs)
        self.assertIn ((2, 3), pairs)
        self.assertIn ((2, 4), pairs)
        self.assertIn ((2, 1), pairs)
        self.assertIn ((5, 4), pairs)
        self.assertIn ((5, 1), pairs)
        self.assertIn ((5, 3), pairs)
        self.assertIn ((4, 2), pairs)
        self.assertIn ((4, 5), pairs)
        self.assertIn ((4, 3), pairs)

        pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = False, d_lower = 0, d_upper = np.inf)

        self.assertEqual (len (pairs), 9)
        self.assertIn ((0, 1), pairs)
        self.assertIn ((0, 2), pairs)
        self.assertIn ((0, 3), pairs)
        self.assertIn ((0, 5), pairs)
        self.assertIn ((0, 4), pairs)
        self.assertIn ((1, 3), pairs)
        self.assertIn ((2, 3), pairs)
        self.assertIn ((5, 3), pairs)
        self.assertIn ((4, 3), pairs)

    def test_d (self):
        cretention = retention_cls()

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2),
                                 (("M7", "B"), 3)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {0: [],
                       1: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (4, 5), (4, 2), (5, 3), (5, 6)],
                       2: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (4, 5), (4, 2), (5, 3), (5, 6),
                           (0, 2), (0, 5), (1, 6), (1, 3), (4, 6), (4, 3)],
                       3: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (4, 5), (4, 2), (5, 3), (5, 6),
                           (0, 2), (0, 5), (1, 6), (1, 3), (4, 6), (4, 3),
                           (0, 3), (0, 6)]}

        for allow_overlap in [True, False]:
            for d in d_pairs_ref.keys():
                pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = allow_overlap,
                                                    d_lower = 0, d_upper = d)

                self.assertEqual (len (pairs), len (d_pairs_ref[d]))

                for pair in  d_pairs_ref[d]:
                    self.assertIn (pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {0: [],
                       1: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 4), (2, 1), (5, 4), (5, 3), (5, 1), (4, 2), (4, 5)],
                       2: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 4), (2, 1), (5, 4), (5, 3), (5, 1), (4, 2), (4, 5),
                           (0, 2), (0, 5), (1, 3), (4, 3)],
                       3: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 4), (2, 1), (5, 4), (5, 3), (5, 1), (4, 2), (4, 5),
                           (0, 2), (0, 5), (1, 3), (4, 3),
                           (0, 3)]}

        for d in d_pairs_ref.keys():
            pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = True,
                                                d_lower = 0, d_upper = d)

            self.assertEqual (len (pairs), len (d_pairs_ref[d]))

            for pair in  d_pairs_ref[d]:
                self.assertIn (pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {0: [],
                       1: [(0, 1), (0, 4), (2, 3), (5, 3)],
                       2: [(0, 1), (0, 4), (2, 3), (5, 3),
                           (0, 2), (0, 5), (1, 3), (4, 3)],
                       3: [(0, 1), (0, 4), (2, 3), (5, 3),
                           (0, 2), (0, 5), (1, 3), (4, 3),
                           (0, 3)]}

        for d in d_pairs_ref.keys():
            pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = False, d_lower = 0,
                                                d_upper = d)

            self.assertEqual (len (pairs), len (d_pairs_ref[d]))

            for pair in  d_pairs_ref[d]:
                self.assertIn (pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {4: [],
                       3: [(0, 3)],
                       2: [(0, 2), (0, 3), (0, 5), (1, 3), (4, 3)],
                       1: [(0, 1), (0, 4), (2, 3), (5, 3),
                           (0, 2), (0, 5), (1, 3), (4, 3),
                           (0, 3)],
                       0: [(0, 1), (0, 4), (2, 3), (5, 3),
                           (0, 2), (0, 5), (1, 3), (4, 3),
                           (0, 3)]}

        for d in d_pairs_ref.keys():
            pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = False, d_lower = d,
                                                d_upper = np.inf)

            self.assertEqual (len (pairs), len (d_pairs_ref[d]))

            for pair in  d_pairs_ref[d]:
                self.assertIn (pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2), (("M7", "B"), 1.5)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {4: [],
                       3: [(0, 3)],
                       2: [(0, 3), (0, 5), (0, 2), (0, 6),
                           (1, 3),
                           (4, 3),
                           (6, 3)],
                       1: [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                           (1, 2), (1, 3), (1, 5), (1, 6),
                           (2, 3),
                           (4, 2), (4, 3), (4, 5), (4, 6),
                           (5, 3),
                           (6, 2), (6, 3), (6, 5)],
                       0: [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                           (1, 2), (1, 3), (1, 5), (1, 6),
                           (2, 3),
                           (4, 2), (4, 3), (4, 5), (4, 6),
                           (5, 3),
                           (6, 2), (6, 3), (6, 5)]}

        for allow_overlap in [True, False]:
            for d in d_pairs_ref.keys():
                pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = allow_overlap, d_lower = d,
                                                    d_upper = np.inf)

                self.assertEqual (len (pairs), len (d_pairs_ref[d]))

                for pair in  d_pairs_ref[d]:
                    self.assertIn (pair, pairs)

    def test_ireversed (self):
        cretention = retention_cls()

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2),
                                 (("M7", "B"), 3)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph (ireverse = 0)
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        pairs_ref = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (0, 2), (1, 3), (4, 6), (0, 3)]
        pairs_notin_ref = [(1, 6), (1, 5), (0, 5), (0, 4)]

        for allow_overlap in [True, False]:
            pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = allow_overlap,
                                                d_lower = 0, d_upper = np.inf)

            self.assertEqual (len (pairs), len (pairs_ref))

            for pair in pairs_ref:
                self.assertIn (pair, pairs)

            for pair in pairs_notin_ref:
                self.assertNotIn (pair, pairs)

    def test_equal_to_simple_function_in_single_system_case (self):
        cretention = retention_cls()

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 10), (("M2", "A"), 4), (("M3", "A"), 6), (("M4", "A"), 8),
                                 (("M5", "A"), 2)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {0: [],
                       1: [(4, 1), (1, 2), (2, 3), (3, 0)],
                       2: [(4, 1), (1, 2), (2, 3), (3, 0),
                           (4, 2), (1, 3), (2, 0)],
                       3: [(4, 1), (1, 2), (2, 3), (3, 0),
                           (4, 2), (1, 3), (2, 0),
                           (4, 3), (1, 0)],
                       4: [(4, 1), (1, 2), (2, 3), (3, 0),
                           (4, 2), (1, 3), (2, 0),
                           (4, 3), (1, 0),
                           (4, 0)]}

        for d in d_pairs_ref.keys():
            pairs_og = get_pairs_from_order_graph (cretention, keys, allow_overlap = True,
                                                   d_lower = 0, d_upper = d)
            pairs = get_pairs_single_system (list (d_target.values()), d_lower = 0, d_upper = d)

            self.assertEqual (len (pairs_og), len (d_pairs_ref[d]))
            self.assertEqual (len (pairs), len (d_pairs_ref[d]))

            for pair in d_pairs_ref[d]:
                self.assertIn (pair, pairs_og)
                self.assertIn (pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 10), (("M2", "A"), 4), (("M3", "A"), 6), (("M4", "A"), 8),
                                 (("M5", "A"), 2)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {5: [],
                       4: [(4, 0)],
                       3: [(4, 0), (4, 3), (1, 0)],
                       2: [(4, 0), (4, 3), (1, 0), (4, 2), (1, 3), (2, 0)],
                       1: [(4, 0), (4, 3), (1, 0), (4, 2), (1, 3), (2, 0), (4, 1), (1, 2), (2, 3), (3, 0)],
                       0: [(4, 0), (4, 3), (1, 0), (4, 2), (1, 3), (2, 0), (4, 1), (1, 2), (2, 3), (3, 0)]}

        for d in d_pairs_ref.keys():
            pairs_og = get_pairs_from_order_graph (cretention, keys, allow_overlap = True,
                                                   d_lower = d, d_upper = np.inf)
            pairs = get_pairs_single_system (list (d_target.values()), d_lower = d, d_upper = np.inf)

            self.assertEqual (len (pairs_og), len (d_pairs_ref[d]))
            self.assertEqual (len (pairs), len (d_pairs_ref[d]))

            for pair in d_pairs_ref[d]:
                self.assertIn (pair, pairs_og)
                self.assertIn (pair, pairs)

    def test_equal_to_simple_function_in_multiple_system_case (self):
        cretention = retention_cls()

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 5), (("M2", "A"), 2), (("M3", "A"), 3), (("M4", "A"), 4),
                                 (("M5", "A"), 1),
                                 (("M5", "B"), 1), (("M9", "B"), 10), (("M7", "B"), 5), (("M1", "B"), 12)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph (ireverse = 0)
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {0: [],
                       1: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8)],
                       2: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8),
                           (4, 2), (1, 3), (2, 0), (5, 6), (7, 8)],
                       3: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8),
                           (4, 2), (1, 3), (2, 0), (5, 6), (7, 8),
                           (4, 3), (1, 0), (5, 8)],
                       4: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8),
                           (4, 2), (1, 3), (2, 0), (5, 6), (7, 8),
                           (4, 3), (1, 0), (5, 8),
                           (4, 0)]}
        for d in d_pairs_ref.keys():
            pairs_og = get_pairs_from_order_graph (cretention, keys, allow_overlap = True,
                                                   d_lower = 0, d_upper= d)
            m_target = np.array ([list (d_target.values()), [1, 1, 1, 1, 1, 2, 2, 2, 2]]).T
            pairs = get_pairs_multiple_systems (m_target, d_lower = 0, d_upper= d)

            self.assertEqual (len (pairs_og), len (d_pairs_ref[d]))
            self.assertEqual (len (pairs), len (d_pairs_ref[d]))

            for pair in d_pairs_ref[d]:
                self.assertIn (pair, pairs_og)
                self.assertIn (pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict ([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                 (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2), (("M7", "B"), 1.5)])
        keys = list (d_target.keys())

        cretention.load_data_from_target (d_target)
        cretention.make_digraph (ireverse = False)
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {4: [],
                       3: [(0, 3)],
                       2: [(0, 3), (0, 2), (1, 3), (4, 5)],
                       1: [(0, 1), (0, 2), (0, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (6, 5)],
                       0: [(0, 1), (0, 2), (0, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (6, 5)]}

        for allow_overlap in [True, False]:
            for d in d_pairs_ref.keys():
                pairs = get_pairs_from_order_graph (cretention, keys, allow_overlap = allow_overlap, d_lower = d,
                                                    d_upper = np.inf)

                m_target = np.array ([list (d_target.values()), [1, 1, 1, 1, 2, 2, 2]]).T
                pairs_sf = get_pairs_multiple_systems (m_target, d_lower = d, d_upper = np.inf)

                self.assertEqual (len (pairs), len (d_pairs_ref[d]))
                self.assertEqual (len (pairs_sf), len (d_pairs_ref[d]))

                for pair in  d_pairs_ref[d]:
                    self.assertIn (pair, pairs)
                    self.assertIn (pair, pairs_sf)

class Test_tanimoto_kernel (unittest.TestCase):
    def test_vector_implementation (self):
        A = np.array ([[0, 1, 0, 1, 1, 1],
                       [0, 0, 1, 1, 1, 0],
                       [0, 1, 0, 1, 1, 1],
                       [1, 0, 0, 1, 0, 0]])

        for i, j in it.combinations_with_replacement (range (A.shape[0]), 2):
            self.assertEqual (tanimoto_kernel (A[i], A[j]), tanimoto_kernel (A[j], A[i]))
            self.assertEqual (tanimoto_kernel (A[i], A[j]), np.sum (A[i] & A[j]) / np.sum (A[i] | A[j]))

    def test_compare_vector_and_matrix_implementation (self):
        A = np.array([[0, 1, 0, 1, 1, 1],
                      [0, 0, 1, 1, 1, 0],
                      [0, 1, 0, 1, 1, 1],
                      [1, 0, 0, 1, 0, 0]])
        K_tan = tanimoto_kernel_mat (A)
        self.assertEqual (K_tan.shape, (4, 4))

        for i, j in it.combinations_with_replacement (range (A.shape[0]), 2):
            self.assertEqual (K_tan[i, j], tanimoto_kernel (A[i], A[j]))
            self.assertEqual (K_tan[i, j], K_tan[j, i])

        # Different matrices
        B = np.array([[0, 0, 1, 1, 1, 1],
                      [1, 0, 1, 1, 0, 0],
                      [1, 1, 0, 1, 1, 1]])

        K_tan = tanimoto_kernel_mat (A, B)
        self.assertEqual (K_tan.shape, (4, 3))

        for i, j in it.product (range (A.shape[0]), range (B.shape[0])):
            self.assertEqual (K_tan[i, j], tanimoto_kernel (A[i], B[j]))

class Test_minmax_kernel (unittest.TestCase):
    def test_compare_vector_and_matrix_implementation (self):
        A = np.array([[0, 7, 0, 1, 3, 1],
                      [0, 0, 1, 4, 5, 0],
                      [0, 8, 1, 9, 1, 2],
                      [7, 0, 0, 2, 0, 0]])

        K_mm = minmax_kernel_mat (A)

        self.assertEqual (K_mm.shape, (4, 4))

        for i, j in it.combinations_with_replacement (range (A.shape[0]), 2):
            self.assertEqual (K_mm[i, j], minmax_kernel (A[i], A[j]))
            self.assertEqual (K_mm[i, j], K_mm[j, i])

        # Different matrices
        B = np.array([[0, 0, 6, 1, 1, 9],
                      [1, 0, 1, 3, 0, 0],
                      [2, 1, 0, 1, 1, 4]])

        K_mm = minmax_kernel_mat (A, B)
        self.assertEqual (K_mm.shape, (4, 3))

        for i, j in it.product (range (A.shape[0]), range (B.shape[0])):
            self.assertEqual (K_mm[i, j], minmax_kernel (A[i], B[j]))

if __name__ == '__main__':
    unittest.main()
