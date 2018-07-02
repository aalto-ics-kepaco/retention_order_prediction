import unittest
import numpy as np
import copy

from evaluation_scenarios_cls import build_candidate_structure, shortest_path, _weight_func_max
from evaluation_scenarios_cls import shortest_path_exclude_candidates
from evaluation_scenarios_cls import perform_reranking_of_candidates

from rank_svm_cls import tanimoto_kernel

class DummyRankingModel:
    def __init__(self, kernel):
        self.kernel = kernel
    def map_values(self, X):
        return X.shape[1] # For our toy data this always returns 2!

class DummyRankingModel2:
    def __init__(self, kernel):
        self.kernel = kernel
    def map_values(self, X):
        return [-np.sum(X)]

def make_candidate_data():
    base_dir = "/home/bach/Documents/studies/doctoral/projects/rt_prediction_ranksvm/"
    input_dir_candidates = base_dir + "/data/processed/Impact/toy_candidates/"

    model = {"ranking_model": DummyRankingModel2(tanimoto_kernel),
             "kernel_params": {},
             "training_data": np.array([]),
             "predictor": ["maccs"]}

    cand_data = build_candidate_structure(
        input_dir_candidates=input_dir_candidates, model=model, n_jobs=2, debug=True)
    return cand_data

class Test_reranking_of_candidates(unittest.TestCase):
    def test_reranking(self):
        cand_data = make_candidate_data()

        topk_accs, paths, lengths = perform_reranking_of_candidates(
            cand_data=cand_data, weight_function=_weight_func_max,
            cut_off_n_cand=100, topk=25, **{"D": 0.0, "use_sign": False, "epsilon_rt": 0.0, "use_log": False})

        self.assertEqual(len(topk_accs), 3)
        self.assertEqual(topk_accs[0], 50)
        self.assertEqual(topk_accs[1], 75)
        self.assertEqual(topk_accs[2], 100)

        self.assertTrue(paths[0] == [0, 0, 0, 0])
        self.assertTrue(paths[1] == [1, 1, 1, 1])
        self.assertTrue(paths[2] == [2, 2, 2])

class Test_shortest_path_algorithms(unittest.TestCase):
    def test_shortest_path(self):
        cand_data = make_candidate_data()
        self.assertEqual(len(cand_data), 4)

        # Put zero weight on the ranking scores
        cand_on_shortest_path = ["C_1_1st", "C_4_1st", "C_7_1st", "C_8_1st"]
        path, length = shortest_path(cand_data, _weight_func_max, cut_off_n_cand=np.inf, check_input=True,
                                     **{"D": 0, "use_sign": False, "epsilon_rt": 0.0, "use_log": False})

        self.assertEqual(len(path), len(cand_data))
        length_ref = 0.0
        for t, node in enumerate(path):
            self.assertEqual(node, 0)
            self.assertEqual(cand_data[t]["inchis"][node], cand_on_shortest_path[t])
            length_ref += -cand_data[t]["iokrscores"][node]
        self.assertEqual(length_ref, length)

        # Put low weight on the ranking scores (which does not effect the shortest path)
        cand_on_shortest_path = ["C_1_1st", "C_4_1st", "C_7_1st", "C_8_1st"]
        path, length = shortest_path(cand_data, _weight_func_max, cut_off_n_cand=np.inf, check_input=True,
                                     **{"D": 1, "use_sign": False, "epsilon_rt": 0.0, "use_log": False})

        self.assertEqual(len(path), len(cand_data))
        length_ref = 0.0
        for t, node in enumerate(path):
            self.assertEqual(cand_data[t]["inchis"][node], cand_on_shortest_path[t])
            length_ref += -cand_data[t]["iokrscores"][node]
        length_ref += 1 # Order violation when going from S7 to S8
        self.assertEqual(length_ref, length)

        # Put higher weight on the ranking scores
        # Weight on the order consistence is large enough to force the shortest path to
        # go through the second candidate of S8.
        cand_on_shortest_path = ["C_1_1st", "C_4_1st", "C_7_1st", "C_8_2nd"]
        path, length = shortest_path(cand_data, _weight_func_max, cut_off_n_cand=np.inf, check_input=True,
                                     **{"D": 3, "use_sign": False, "epsilon_rt": 0.0, "use_log": False})

        self.assertEqual(len(path), len(cand_data))
        length_ref = 0.0
        for t, node in enumerate(path):
            self.assertEqual(cand_data[t]["inchis"][node], cand_on_shortest_path[t])
            length_ref += -cand_data[t]["iokrscores"][node]
        self.assertEqual(length_ref, length)

    def test_shortest_path_with_blocked_candidates(self):
        cand_data = make_candidate_data()
        self.assertEqual(len(cand_data), 4)

        for t, _ in enumerate(cand_data):
            n_cand_t = len(cand_data[t]["iokrscores"])
            cand_data[t]["is_blocked"] = np.zeros(n_cand_t, dtype="bool")

        # Block all candidates with the highest score:
        cand_data_c = copy.deepcopy(cand_data)
        for t, _ in enumerate(cand_data_c):
            cand_data_c[t]["is_blocked"][0] = True

        # Put zero weight on the ranking scores
        cand_on_shortest_path = ["C_1_2nd", "C_4_2nd", "C_7_2nd", "C_8_2nd"]
        path, length = shortest_path_exclude_candidates(
            cand_data_c, _weight_func_max, cut_off_n_cand=np.inf, check_input=True, exclude_blocked_candidates=True,
            **{"D": 0, "use_sign": False, "epsilon_rt": 0.0, "use_log": False})

        self.assertEqual(len(path), len(cand_data_c))
        length_ref = 0.0
        for t, node in enumerate(path):
            self.assertEqual(node, 1)
            self.assertEqual(cand_data_c[t]["inchis"][node], cand_on_shortest_path[t])
            length_ref += -cand_data_c[t]["iokrscores"][node]
        self.assertEqual(length_ref, length)

        # Block only two candidates with the highest scores:
        cand_data_c = copy.deepcopy(cand_data)
        cand_data_c[1]["is_blocked"][0] = True
        cand_data_c[2]["is_blocked"][0] = True

        # Put zero weight on the ranking scores
        cand_on_shortest_path = ["C_1_1st", "C_4_2nd", "C_7_2nd", "C_8_1st"]
        path, length = shortest_path_exclude_candidates(
            cand_data_c, _weight_func_max, cut_off_n_cand=np.inf, check_input=True, exclude_blocked_candidates=True,
            **{"D": 0, "use_sign": False, "epsilon_rt": 0.0, "use_log": False})

        self.assertEqual(len(path), len(cand_data_c))
        length_ref = 0.0
        for t, node in enumerate(path):
            if t == 0 or t == 3:
                self.assertEqual(node, 0)
            else:
                self.assertEqual(node, 1)
            self.assertEqual(cand_data_c[t]["inchis"][node], cand_on_shortest_path[t])
            length_ref += -cand_data_c[t]["iokrscores"][node]
        self.assertEqual(length_ref, length)

        # Put higher weight on the ranking scores
        # Weight on the order consistence is large enough to force the shortest path to
        # go through the second candidate of S8.

        # Block only two candidates with the highest scores:
        cand_data_c = copy.deepcopy(cand_data)
        cand_data_c[3]["is_blocked"][1] = True

        cand_on_shortest_path = ["C_1_1st", "C_4_1st", "C_7_1st", "C_8_1st"]
        path, length = shortest_path_exclude_candidates(
            cand_data_c, _weight_func_max, cut_off_n_cand=np.inf, check_input=True, exclude_blocked_candidates=True,
            **{"D": 3, "use_sign": False, "epsilon_rt": 0.0, "use_log": False})

        self.assertEqual(len(path), len(cand_data_c))
        length_ref = 0.0
        for t, node in enumerate(path):
            self.assertEqual(node, 0)
            self.assertEqual(cand_data_c[t]["inchis"][node], cand_on_shortest_path[t])
            length_ref += -cand_data_c[t]["iokrscores"][node]
        length_ref += 3
        self.assertEqual(length_ref, length)

if __name__ == '__main__':
    unittest.main()
