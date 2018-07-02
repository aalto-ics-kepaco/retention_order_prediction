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
