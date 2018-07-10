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

from retention_cls import retention_cls

class Test_load_data_from_target (unittest.TestCase):
    def test_simplecases(self):
        cretention = retention_cls()

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2,
                    ("M1", "B"): 2, ("M2", "B"): 1}

        cretention.load_data_from_target (d_target)

        self.assertEqual (len (cretention.lrows), 2)
        self.assertIn (["M1", "M2", 1, "A"], cretention.lrows)
        self.assertIn (["M2", "M1", 1, "B"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target (d_target)

        self.assertEqual (len (cretention.lrows), 2)
        self.assertIn (["M1", "M2", 1, "A"], cretention.lrows)
        self.assertIn (["M1", "M2", 1, "B"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2,  ("M3", "A"): 3, ("M4", "A"): 4,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target (d_target)

        self.assertEqual (len (cretention.lrows), 4)
        self.assertIn (["M1", "M2", 1, "A"], cretention.lrows)
        self.assertIn (["M2", "M3", 1, "A"], cretention.lrows)
        self.assertIn (["M3", "M4", 1, "A"], cretention.lrows)
        self.assertIn (["M1", "M2", 1, "B"], cretention.lrows)

    def test_bordercases (self):
        cretention = retention_cls()

        # ----------------------------------------------------
        d_target = {}

        cretention.load_data_from_target (d_target)

        self.assertEqual (cretention.lrows, [])


        # ----------------------------------------------------
        d_target = {("M2", "B"): 7}

        cretention.load_data_from_target (d_target)

        self.assertEqual (cretention.lrows, [])

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "B"): 7}

        cretention.load_data_from_target (d_target)

        self.assertEqual (cretention.lrows, [])

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 3, ("M2", "B"): 7, ("M3", "B"): 19}

        cretention.load_data_from_target (d_target, linclude_collection = [])

        self.assertEqual (cretention.lrows, [])


        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 3, ("M2", "B"): 7, ("M3", "B"): 19}

        cretention.load_data_from_target (d_target, linclude_node = [])

        self.assertEqual (cretention.lrows, [])

    def test_complexcases (self):
        """
        TEST: Nodes with the same target value are handled correctly.
        """
        cretention = retention_cls()

        # ----------------------------------------------------
        d_target = {("M1", "A"): 7, ("M2", "A"): 7, ("M3", "A"): 3, ("M4", "A"): 8,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target (d_target)

        self.assertEqual (len (cretention.lrows), 7)
        self.assertIn (["M3", "M1", 1, "A"], cretention.lrows)
        self.assertIn (["M3", "M2", 1, "A"], cretention.lrows)
        self.assertIn (["M1", "M2", 0, "A"], cretention.lrows)
        self.assertIn (["M2", "M1", 0, "A"], cretention.lrows)
        self.assertIn (["M1", "M4", 1, "A"], cretention.lrows)
        self.assertIn (["M2", "M4", 1, "A"], cretention.lrows)
        self.assertIn (["M1", "M2", 1, "B"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 7, ("M2", "A"): 7, ("M3", "A"): 7}

        cretention.load_data_from_target (d_target)

        self.assertEqual (len (cretention.lrows), 6)
        self.assertIn (["M1", "M2", 0, "A"], cretention.lrows)
        self.assertIn (["M1", "M3", 0, "A"], cretention.lrows)
        self.assertIn (["M2", "M1", 0, "A"], cretention.lrows)
        self.assertIn (["M2", "M3", 0, "A"], cretention.lrows)
        self.assertIn (["M3", "M1", 0, "A"], cretention.lrows)
        self.assertIn (["M3", "M2", 0, "A"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 7, ("M2", "A"): 7, ("M3", "A"): 7, ("M4", "A"): 1}

        cretention.load_data_from_target (d_target)

        self.assertEqual (len (cretention.lrows), 9)
        self.assertIn (["M4", "M1", 1, "A"], cretention.lrows)
        self.assertIn (["M4", "M2", 1, "A"], cretention.lrows)
        self.assertIn (["M4", "M3", 1, "A"], cretention.lrows)
        self.assertIn (["M1", "M2", 0, "A"], cretention.lrows)
        self.assertIn (["M1", "M3", 0, "A"], cretention.lrows)
        self.assertIn (["M2", "M1", 0, "A"], cretention.lrows)
        self.assertIn (["M2", "M3", 0, "A"], cretention.lrows)
        self.assertIn (["M3", "M1", 0, "A"], cretention.lrows)
        self.assertIn (["M3", "M2", 0, "A"], cretention.lrows)

    def test_exclude_nodes_and_collections (self):
        """
        TEST: Nodes and collections are ignored if specified.
        """
        cretention = retention_cls()

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2, ("M3", "A"): 3, ("M4", "A"): 4,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target (d_target, linclude_collection = "B")

        self.assertEqual (len (cretention.lrows), 1)
        self.assertIn (["M1", "M2", 1, "B"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2, ("M3", "A"): 3, ("M4", "A"): 4,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target (d_target, linclude_collection = "A")

        self.assertEqual (len (cretention.lrows), 3)
        self.assertIn (["M1", "M2", 1, "A"], cretention.lrows)
        self.assertIn (["M2", "M3", 1, "A"], cretention.lrows)
        self.assertIn (["M3", "M4", 1, "A"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2, ("M3", "A"): 3, ("M4", "A"): 4,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target (d_target, linclude_collection = "A",
                                          linclude_node = ["M1", "M4"])

        self.assertEqual (len (cretention.lrows), 1)
        self.assertIn (["M1", "M4", 1, "A"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 7, ("M2", "A"): 7, ("M3", "A"): 7, ("M4", "A"): 1}

        cretention.load_data_from_target (d_target, linclude_node = ["M1", "M2", "M4"])

        self.assertEqual (len (cretention.lrows), 4)
        self.assertIn (["M4", "M1", 1, "A"], cretention.lrows)
        self.assertIn (["M4", "M2", 1, "A"], cretention.lrows)
        self.assertIn (["M1", "M2", 0, "A"], cretention.lrows)
        self.assertIn (["M2", "M1", 0, "A"], cretention.lrows)

if __name__ == '__main__':
    unittest.main()
