import unittest
import numpy as np

from helper_cls import sample_perc_from_list, _sample_perc_from_list
from helper_cls import get_statistic_about_concordant_and_discordant_pairs


class Test_concordant_and_discordant_statistics(unittest.TestCase):


    def test_corner_cases(self):
        self.assertEqual(get_statistic_about_concordant_and_discordant_pairs([],[]), {})
        self.assertEqual(get_statistic_about_concordant_and_discordant_pairs([],[("A", "S1"), ("B", "S1")]), {})

    def test_for_single_system(self):
        keys = [("A", "S1"), ("B", "S1"), ("C", "S1"), ("D", "S1")]
        pairs = [(0, 1), (1, 2), (3, 2), (1, 3), (0, 2)]

        stats = get_statistic_about_concordant_and_discordant_pairs(pairs, keys)
        self.assertEqual(len(stats), 5)

        self.assertIn(("A", "B"), stats.keys())
        self.assertIn(("B", "C"), stats.keys())
        self.assertIn(("D", "C"), stats.keys())
        self.assertIn(("B", "D"), stats.keys())
        self.assertIn(("A", "C"), stats.keys())

        self.assertEqual(stats[("A", "B")]["#Pij>"], 1)
        self.assertEqual(stats[("A", "B")]["#Pij<"], 0)
        self.assertEqual(stats[("B", "C")]["#Pij>"], 1)
        self.assertEqual(stats[("B", "C")]["#Pij<"], 0)
        self.assertEqual(stats[("D", "C")]["#Pij>"], 1)
        self.assertEqual(stats[("D", "C")]["#Pij<"], 0)
        self.assertEqual(stats[("B", "D")]["#Pij>"], 1)
        self.assertEqual(stats[("B", "D")]["#Pij<"], 0)
        self.assertEqual(stats[("A", "C")]["#Pij>"], 1)
        self.assertEqual(stats[("A", "C")]["#Pij<"], 0)

        self.assertEqual(stats[("A", "B")]["Pij"], {(0, 1)})
        self.assertEqual(stats[("B", "C")]["Pij"], {(1, 2)})
        self.assertEqual(stats[("D", "C")]["Pij"], {(3, 2)})
        self.assertEqual(stats[("B", "D")]["Pij"], {(1, 3)})
        self.assertEqual(stats[("A", "C")]["Pij"], {(0, 2)})

    def test_for_multiple_systems(self):
        keys_S1 = [("A", "S1"), ("B", "S1"), ("C", "S1"), ("D", "S1")] # largest index 3
        pairs_S1 = [(0, 1), (1, 2), (3, 2), (1, 3), (0, 2)]

        keys_S2 = [("A", "S2"), ("B", "S2"), ("E", "S2"), ("D", "S2")]
        pairs_S2 = [(4, 5), (7, 5), (6, 7)]

        stats = get_statistic_about_concordant_and_discordant_pairs(pairs_S1 + pairs_S2, keys_S1 + keys_S2)

        self.assertEqual(len(stats), 6)

        self.assertIn(("A", "B"), stats.keys())
        self.assertIn(("B", "C"), stats.keys())
        self.assertIn(("D", "C"), stats.keys())
        self.assertIn(("B", "D"), stats.keys())
        self.assertIn(("A", "C"), stats.keys())
        self.assertIn(("E", "D"), stats.keys())

        self.assertEqual(stats[("A", "B")]["#Pij>"], 2)
        self.assertEqual(stats[("A", "B")]["#Pij<"], 0)
        self.assertEqual(stats[("B", "C")]["#Pij>"], 1)
        self.assertEqual(stats[("B", "C")]["#Pij<"], 0)
        self.assertEqual(stats[("D", "C")]["#Pij>"], 1)
        self.assertEqual(stats[("D", "C")]["#Pij<"], 0)
        self.assertEqual(stats[("B", "D")]["#Pij>"], 1)
        self.assertEqual(stats[("B", "D")]["#Pij<"], 1)
        self.assertEqual(stats[("A", "C")]["#Pij>"], 1)
        self.assertEqual(stats[("A", "C")]["#Pij<"], 0)
        self.assertEqual(stats[("E", "D")]["#Pij>"], 1)
        self.assertEqual(stats[("E", "D")]["#Pij<"], 0)

        self.assertEqual(stats[("A", "B")]["Pij"], {(0, 1), (4, 5)})
        self.assertEqual(stats[("B", "C")]["Pij"], {(1, 2)})
        self.assertEqual(stats[("D", "C")]["Pij"], {(3, 2)})
        self.assertEqual(stats[("B", "D")]["Pij"], {(1, 3), (7, 5)})
        self.assertEqual(stats[("A", "C")]["Pij"], {(0, 2)})
        self.assertEqual(stats[("E", "D")]["Pij"], {(6, 7)})

        stats = get_statistic_about_concordant_and_discordant_pairs(pairs_S2 + pairs_S1, keys_S1 + keys_S2)

        self.assertEqual(len(stats), 6)

        self.assertIn(("A", "B"), stats.keys())
        self.assertIn(("B", "C"), stats.keys())
        self.assertIn(("D", "C"), stats.keys())
        self.assertIn(("D", "B"), stats.keys())
        self.assertIn(("A", "C"), stats.keys())
        self.assertIn(("E", "D"), stats.keys())

        self.assertEqual(stats[("A", "B")]["#Pij>"], 2)
        self.assertEqual(stats[("A", "B")]["#Pij<"], 0)
        self.assertEqual(stats[("B", "C")]["#Pij>"], 1)
        self.assertEqual(stats[("B", "C")]["#Pij<"], 0)
        self.assertEqual(stats[("D", "C")]["#Pij>"], 1)
        self.assertEqual(stats[("D", "C")]["#Pij<"], 0)
        self.assertEqual(stats[("D", "B")]["#Pij>"], 1)
        self.assertEqual(stats[("D", "B")]["#Pij<"], 1)
        self.assertEqual(stats[("A", "C")]["#Pij>"], 1)
        self.assertEqual(stats[("A", "C")]["#Pij<"], 0)
        self.assertEqual(stats[("E", "D")]["#Pij>"], 1)
        self.assertEqual(stats[("E", "D")]["#Pij<"], 0)

        self.assertEqual(stats[("A", "B")]["Pij"], {(0, 1), (4, 5)})
        self.assertEqual(stats[("B", "C")]["Pij"], {(1, 2)})
        self.assertEqual(stats[("D", "C")]["Pij"], {(3, 2)})
        self.assertEqual(stats[("D", "B")]["Pij"], {(1, 3), (7, 5)})
        self.assertEqual(stats[("A", "C")]["Pij"], {(0, 2)})
        self.assertEqual(stats[("E", "D")]["Pij"], {(6, 7)})

class Test_random_subsampling_of_lists(unittest.TestCase):
    def test_produce_some_subsamples(self):
        lst = list(range(55))
        n = len(lst)

        # Sample 100%
        lst_sub = _sample_perc_from_list(lst)
        self.assertEqual(lst, lst_sub)

        # Sample 0%
        for algo in ["random", "cum_rand"]:
            lst_sub = _sample_perc_from_list(lst, perc=0, algorithm=algo)
            self.assertEqual(lst_sub, [])

        # Sample
        for algo in ["random", "cum_rand"]:
            for perc in range(0, 101, 10):
                lst_sub = _sample_perc_from_list(lst, perc=perc, algorithm=algo)

                # Subsample is still sorted (input was sorted)
                self.assertTrue(all(lst_sub[i] <= lst_sub[i+1] for i in range(len(lst_sub)-1)))

                print("[%s] perc=%f, real perc=%f, delta perc=%f"
                      % (algo, perc, 100 * len(lst_sub) / n, perc - 100 * len(lst_sub) / n))

        # ----------------------------------------------------
        lst = list(range(100))

        for perc in range(0, 101, 10):
            lst_sub = _sample_perc_from_list(lst, perc=perc, algorithm="cum_rand")

            # Subsample is still sorted (input was sorted)
            self.assertTrue(all(lst_sub[i] <= lst_sub[i+1] for i in range(len(lst_sub)-1)))
            self.assertTrue(len(lst_sub) == perc)

    def test_produce_some_subsamples_considering_a_certain_system(self):
        for _ in range(25):
            lst = []
            sys_A = sys_B = sys_C = 0
            for idx in range(90):
                resample = True

                while resample:
                    rnd = np.random.randint(0, 3)

                    if rnd == 0 and sys_A < 30:
                        lst.append((idx, "A"))
                        resample = False
                        sys_A += 1
                    elif rnd == 1 and sys_B < 30:
                        lst.append((idx, "B"))
                        resample = False
                        sys_B += 1
                    elif rnd == 2 and sys_C < 30:
                        lst.append((idx, "C"))
                        resample = False
                        sys_C += 1
                    else:
                        RuntimeError("Ups, I though we sample from [low,high)?!")

            # Sample 100%
            for sys in [None, "A", "B", "C"]:
                lst_sub = sample_perc_from_list(lst, tsystem=sys)
                self.assertEqual(lst_sub, lst)

            # Sample 0%
            lst_sub = sample_perc_from_list(lst, tsystem=None, perc=0)
            self.assertEqual(lst_sub, [])

            lst_sub = sample_perc_from_list(lst, tsystem="D", perc=0)
            self.assertEqual(lst_sub, lst)

            for sys in ["A", "B", "C"]:
                lst_sub = sample_perc_from_list(lst, tsystem=sys, perc=0)
                self.assertTrue(len(lst_sub) == 60)
                self.assertNotIn(sys, list(zip(*lst_sub))[1])

                for rem_sys in {"A", "B", "C"} - {sys}:
                    self.assertEqual(sum(np.array(list(zip(*lst_sub))[1]) == rem_sys), 30)

            # Sample
            for sys in ["A", "B", "C"]:
                for perc in range(0, 101, 10):
                    lst_sub = sample_perc_from_list(lst, tsystem=sys, perc=perc)

                    # Subsample is still sorted (input was sorted)
                    self.assertTrue(all(lst_sub[i][0] <= lst_sub[i+1][0] for i in range(len(lst_sub)-1)))
                    self.assertEqual(len(lst_sub), 60 + 0.3 * perc)

                    for rem_sys in {"A", "B", "C"} - {sys}:
                        self.assertEqual(sum(np.array(list(zip(*lst_sub))[1]) == rem_sys), 30)

                    self.assertEqual(sum(np.array(list(zip(*lst_sub))[1]) == sys), 0.3 * perc)


if __name__ == '__main__':
    unittest.main()
