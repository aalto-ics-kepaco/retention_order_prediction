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

'''

Implementation of the RankSVM class and helper functions that allow
to create learning pairs from objects target values possible comming
from several queries:

E.g.:
    - Objects are molecules
    - Targets are retention times
    - Queries are chromatographic systems

'''

from pandas import DataFrame
from time import process_time, sleep
from collections import OrderedDict, deque

import numpy as np
import scipy as sp
import cvxpy as cvx
import mosek
import itertools
import warnings
warnings.simplefilter ('ignore', sp.sparse.SparseEfficiencyWarning)
warnings.simplefilter ('always', UserWarning)

## Imports for the implementation of the kernelized rank svm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels, check_pairwise_arrays

# Allow the paralellization
from joblib import Parallel, delayed

def get_pairs_from_order_graph (cretention, keys, allow_overlap, d_lower, d_upper, n_jobs = 1):
    """
    Task: Get the pairs (i,j) with i elutes before j for the learning / prediction process
          given a set of input features with corresponding targets.

    :param cretention: retention_cls object, representation of the order graph.

    :param keys: list, of (mol-id, system)-tuples. This list defines the order,
                 in which the feature vectors are stored in the feature matrix X.
                 The pair indices (i,j) are the indices at which a particular
                 key is found in the 'keys' list.

                 Example:
                 For the difference features the row j of X is subtracted from
                 the row i of X: phi_ij = X[j] - X[i] for pair (i,j).

    :param allow_overlap: binary, should overlaps between the upper and lowers sets be
                          allowed. Those overlaps represent contradictions in the
                          elution orders between different systems.

    :param d_lower: scalar, minimum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param d_upper: scalar, maximum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param n_jobs: scalar, number of jobs to run pair extraction in parallel. (default = 1)

    :return: array-like, shape = (n_pairs,), list of tuples
         [(i,j),(j,k),...], with i elutes before j and j elutes before k, ...

         i,j,k are scalars that correspond to the particular index of the
         targets values (in 'targets'). So we need to keep the target values
         and the feature vectors in the same order!
    """

    # Get upper and lower sets of the nodes
    d_moleculecut = cretention.upper_lower_set_node (cretention.dG)

    # For each node in the graph and its corresponding upper and lower set do:
    pairs = Parallel (n_jobs = n_jobs)(delayed (_get_pairs_for_node)(
        node, ul_set_nodes, keys, cretention.dmolecules_inv,
        cretention.dcollections_inv, allow_overlap, d_lower, d_upper)
                                       for node, ul_set_nodes in d_moleculecut.items())

    # Flatten the result
    pairs = [pair for sublist in pairs for pair in sublist]

    # Due to the symmetry of the upper and lower sets we count every pairwise relation twice:
    #   For examples: Given A->B, B->C, we have C in L(A) ==> (A,C) and A in U(C) ==> (A,C)
    pairs = list (set (pairs))

    return pairs

def _get_pairs_for_node (node, ul_set_node, keys, dmolecules_inv, dcollections_inv, allow_overlap,
                         d_lower, d_upper):
    """
    Task: Get all the pairs for a particular node in the order graph.

    :param node: (mol-id, system)-tuple, node to process

    :param ul_set_node: dictionary, upper and lower set of the current node.

    :param keys: list, of (mol-id, system)-tuples. This list defines the order,
                 in which the feature vectors are stored in the feature matrix X.
                 The pair indices (i,j) are the indices at which a particular
                 key is found in the 'keys' list.

                 Example:
                 For the difference features the row j of X is subtracted from
                 the row i of X: phi_ij = X[j] - X[i] for pair (i,j).

    :param dmolecules_inv:
    :param dcollections_inv:

    :param allow_overlap: binary, should overlaps between the upper and lowers sets be
                          allowed. Those overlaps represent contradictions in the
                          elution orders between different systems.

    :param d_lower: scalar, minimum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param d_upper: scalar, maximum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :return:
    """
    def _find_in (x, l, only_first_occurrence = True):
        """
        Task: Find all indices in the list l where the value l[i] equals x.

        :param x: object, value to search for

        :param l: list, list of objects

        :param only_first_occurrence: boolean, returned list only contains the
                                      index of the first occurrence of the searched
                                      item.

        :return: list of indices, at which x is found in l.
        """
        if only_first_occurrence:
            for i, xi in enumerate (l):
                if xi == x:
                    return [i]
        else:
            return [i for i, xi in enumerate (l) if xi == x]

    pairs = []

    row_i = _find_in ((dmolecules_inv[node[0]], dcollections_inv[node[1]]), keys)
    assert (len (row_i) == 1)

    if allow_overlap:
        u_set_node = ul_set_node[0]
        l_set_node = ul_set_node[1]
    else:
        # Remove overlap between upper and lower set, e.g. due
        # to elution order contradictions.
        u_set_node = OrderedDict ([(key, value) for key, value in ul_set_node[0].items()
                                   if not key in ul_set_node[1].keys()])
        l_set_node = OrderedDict ([(key, value) for key, value in ul_set_node[1].items()
                                   if not key in ul_set_node[0].keys()])

    # for each node in the upper set of i do
    # Upper set: x_j > x_i, j is preferred over i
    for u_set_node, dist in u_set_node.items():
        # Exclude two types of pairs:
        # a) Molecules are the same between two systems
        # b) Their distance in the order graph does not reach
        #    a certain minimum distance
        # c) Their distance in the order graph exceeds a certain
        #    threshold.
        if dist == 0 or dist < d_lower or dist > d_upper:
            continue

        # find its row j in X
        # TODO: Bring this down constant time, by using a dictionary (hash-table).
        row_j = _find_in ((dmolecules_inv[u_set_node[0]], dcollections_inv[u_set_node[1]]), keys)
        assert (len (row_j) == 1)

        # j elutes before i ==> t_j < t_i ==> w^T x_j < w^T x_i
        pairs.append ((row_j[0], row_i[0]))

    # for each node in the lower set of i do
    # Lower set: x_i > x_j, i is preferred over j
    for l_set_node, dist in l_set_node.items():
        # Exclude two types of pairs:
        # a) Molecules are the same between two systems
        # b) Their distance in the order graph does not reach
        #    a certain minimum distance
        # c) Their distance in the order graph exceeds a certain
        #    threshold.
        if dist == 0 or dist < d_lower or dist > d_upper:
            continue

        # find its row j in X
        row_j = _find_in ((dmolecules_inv[l_set_node[0]], dcollections_inv[l_set_node[1]]), keys)
        assert (len (row_j) == 1)

        # i elutes before j ==> t_i < t_j ==> w^T x_i < w^T x_j
        pairs.append ((row_i[0], row_j[0]))

    return pairs

def get_pairs_multiple_systems (targets, d_lower, d_upper):
    """
    Task: Get the pairs (i,j) with i elutes before j for the learning / prediction process
          given a set of input features with corresponding targets.

          This function should be used if several systems are considered, but
          inter-system transitivity is not considered. The construction of a
          retention order graph is not necessary.

          This function transforms the provided targets values into dense ranks
          using 'scipy.stats.rankdata'. In that way it is possible to exclude
          example pairs, those rank differs more than a specified threshold. We
          can in that way reduce the number of learning pairs.

    :param targets: array-like, shape = (n_samples, 2), the targets values, e.g.
                   retention times, for all molecules measured with a set of
                   systems.

                   Example:
                   [rts_i, sys_i,
                    rts_j, sys_j,
                    rts_k, sys_k]

                   sys_i, ... are scalars representing the different systems.

    :param d_lower: scalar, minimum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param d_upper: scalar, maximum rank difference for two examples, to be considered
                    as pair.

    :return: array-like, shape = (n_pairs,), list of tuples
             [(i,j),(j,k),...], with i elutes before j and j elutes before k, ...

             i,j,k are scalars that correspond to the particular index of the
             targets values (in 'targets'). So we need to keep the target values
             and the feature vectors in the same order!
    """
    # Number of feature vectors / samples
    l = len (targets)

    pairs = []

    systems = targets[:, 1]
    ranks = np.zeros (l)

    # Ranks must be calculate per system
    for system in np.unique (systems):
        ranks[systems==system] = sp.stats.rankdata (targets[systems==system, 0], method = "dense")

    for pair in itertools.combinations (range (l), 2):
        i, j = pair

        # Skip pairs not in the same system
        if not (systems[i] == systems[j]):
            continue

        # Skip pairs with the same targets value
        if ranks[i] == ranks[j]:
            continue

        # Skip pairs those rank exceeds the threshold
        if np.abs (ranks[i] - ranks[j]) < d_lower or np.abs (ranks[i] - ranks[j]) > d_upper:
            continue

        if ranks[i] < ranks[j]:
            # i elutes before j ==> t_i < t_j ==> w^T(x_j - x_i) > 0 ==> w^Tx_j > w^Tx_i
            pairs.append ((i, j))
        else:
            # j elutes before i ==> t_j < t_i ==> w^T(x_i - x_j) > 0 ==> w^Tx_i > w^Tx_j
            pairs.append ((j, i))

    return pairs


def get_pairs_single_system (targets, d_lower, d_upper, return_rt_differences = False):
    """
    Task: Get the pairs (i,j) with i elutes before j for the learning / prediction process
          given a set of input features with corresponding targets.

          This function should be used if only one system is considered, i.e.
          we only need to get the learning pairs within one particular system.
          The construction of a retention order graph is not necessary.

          This function transforms the provided targets values into dense ranks
          using 'scipy.stats.rankdata'. In that way it is possible to exclude
          example pairs, those rank differs more than a specified threshold. We
          can in that way reduce the number of learning pairs.

    :param targets: array-like, shape = (n_samples,), the targets values, e.g.
                   retention times, for all molecules measured with a particular
                   system.

                   Example:
                   [rts_i, rts_j, rts_k, ...]

    :param d_lower: scalar, minimum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param d_upper: scalar, maximum rank difference for two examples, to be considered
              as pair.

    :return: array-like, shape = (n_pairs,), list of tuples
             [(i,j),(j,k),...], with i elutes before j and j elutes before k, ...

             i,j,k are scalars that correspond to the particular index of the
             targets values (in 'targets'). So we need to keep the target values
             and the feature vectors in the same order!
    """
    # Number of feature vectors / samples
    l = len (targets)

    pairs = []

    if return_rt_differences:
        rt_diffs = []

    ranks = sp.stats.rankdata (targets, method = "dense")

    for pair in itertools.combinations (range (l), 2):
        i, j = pair

        # Skip pairs with the same targets value
        if ranks[i] == ranks[j]:
            continue

        # Skip pairs those rank exceeds the threshold
        if np.abs (ranks[i] - ranks[j]) < d_lower or np.abs (ranks[i] - ranks[j]) > d_upper:
            continue

        if ranks[i] < ranks[j]:
            # i elutes before j ==> t_i < t_j ==> w^T(x_j - x_i) > 0 ==> w^Tx_j > w^Tx_i
            pairs.append ((i, j))

            if return_rt_differences:
                rt_diffs.append (targets[i] - targets[j])
        else:
            # j elutes before i ==> t_j < t_i ==> w^T(x_i - x_j) > 0 ==> w^Tx_i > w^Tx_j
            pairs.append ((j, i))

            if return_rt_differences:
                rt_diffs.append (targets[j] - targets[i])

    if return_rt_differences:
        return pairs, rt_diffs
    else:
        return pairs

def tanimoto_kernel (x, y):
    """
    Calculates the tanimoto kernel value for two examples represented
    by their feature vector.

    :param x: array-like, shape = (n_features,), example A
    :param y: array-like, shape = (n_features,), example B

    :return: scalar, tanimoto kernel value k(A, B)
    """
    xty = np.dot (x, y)
    xtx = np.dot (x, x)
    yty = np.dot (y, y)

    return xty / (xtx + yty - xty)

def tanimoto_kernel_mat (X, Y = None):
    """
    Calculates the tanimoto kernel value for two sets of examples
    represented by their feature vectors.

    :param X: array-like, shape = (n_samples_A, n_features), examples A
    :param Y: array-like, shape = (n_samples_B, n_features), examples B

    :return: array-like, shape = (n_samples_A, n_samples_B), kernel matrix
             with tanimoto kernel values:

                K[i,j] = k_tan(A_i, B_j)
    """
    X, Y = check_pairwise_arrays (X, Y) # Handle for example Y = None

    n_A = X.shape[0]
    n_B = Y.shape[0]

    P = np.dot (X, Y.T)
    D_A = np.sum (X * X, axis = 1).reshape (-1, 1)
    D_B = np.sum (Y * Y, axis = 1)

    return P / (D_A * np.ones ((1, n_B)) + np.ones ((n_A, 1)) * D_B - P)

def minmax_kernel (x, y):
    """
    Minmax kernel

    :param x: array-like, shape = (n_features,), example A
    :param y: array-like, shape = (n_features,), example B

    :return: scalar, minmax kernel value k(A, B)
    """
    maxkernel = np.maximum (x, y)
    minkernel = np.minimum (x, y)

    return minkernel.sum() / maxkernel.sum()

def minmax_kernel_mat (X, Y = None):
    """
    Calculates the minmax kernel value for two sets of examples
    represented by their feature vectors.

    :param X: array-like, shape = (n_samples_A, n_features), examples A
    :param Y: array-like, shape = (n_samples_B, n_features), examples B

    :return: array-like, shape = (n_samples_A, n_samples_B), kernel matrix
             with minmax kernel values:

                K[i,j] = k_mm(A_i, B_j)

    :source: https://github.com/gmum/pykernels/blob/master/pykernels/regular.py
    """
    X, Y = check_pairwise_arrays (X, Y) # Handle for example Y = None

    minkernel = np.zeros ((X.shape[0], Y.shape[0]))
    maxkernel = np.zeros ((X.shape[0], Y.shape[0]))

    for d in range (X.shape[1]):
        column_1 = X[:, d].reshape (-1, 1)
        column_2 = Y[:, d].reshape (-1, 1)
        minkernel += np.minimum (column_1, column_2.T)
        maxkernel += np.maximum (column_1, column_2.T)

    return minkernel / maxkernel

def load_data (input_dir, system = None, predictor = ["desc"], pred_fn = None,
               fps_as_minusone_and_one = False, verbose = False, rts_as_rank = False):
    """
    Function to load the retention time and molecular features / predictors for a given system
    from the dataset.

    :param input_dir: string, directoy containing the retention times and molecular features / predictors:
        - rts.csv: cvs-file containing the retention times for the different molecules measured
                   with the different systems:

                   E.g.:
                        "inchi","rt","system"
                        "InChI=1S/C10H9N3O/c1-7-11-10...",5.1,"Eawag_XBridgeC18"
                        "InChI=1S/C7H3Br2NO/c8-5-1-4...",8.3,"Eawag_XBridgeC18"
                        ...
                        "InChI=1S/C28H22O6/c29-20-7-2...",33.46,"FEM_long"
                        "InChI=1S/C20H22O8/c21-10-16-17(24)18(25...",27.25,"FEM_long"
                        ...
                        "InChI=1S/C5H4N2O4/c8-3-1-2...",0.7188166667,"LIFE_old"
                        "InChI=1S/C9H10N2O3/c10-7-3-1-6...",1.2945,"LIFE_old"

                   NOTE: Within a single system, each InChI should be unique.
                        ...
          fps.csv: csv-file containing the fingerprints (used as molecular features) for the different
                   molecules measured with the different systems.

                   E.g.:
                        "inchi","maccs_1","maccs_2",...,"maccs_166","pubchem_1","pubchem_2",...
                        "InChI=1S/C10H9N3O/c1-7-11-10...",1,0,...,0,1,0,...
                        "InChI=1S/C28H22O6/c29-20-7-2...",0,1,...,0,1,1,...
                        ...

                   NOTE 1: Here we assume that only the unique molecular structures are containing
                           in the file, i.e. molecules that appear in several systems, only need to be
                           represented ones.
                   NOTE 2: Each column corresponds to a single (in this example) fingerprint bit
                           from a certain fingerprint definition, e.g., MACCS or Pubchem.

    :param system: string, system that should be loaded, if None, than all the systems
        are loaded.

    :param predictor: list of string, containing the predictors that should be loaded:
        E.g.:
            - If predictor = ["maccs","pubchem"], than all columns from 'fps.csv' are extracted
              that contain bits from those two definitions.
            - If predictor = ["maccs"], than only the MACCS columns, i.e., "maccs_1", ..., "maccs_166",
              are loaded.

    :param pred_fn: string, filename of the file containing the predictors (default = None). If the
        its value is None, than the predictor filename is determined automatically:

        - If the predictors == "desc" or any if the predictors is in the list of predefined molecular
          descriptors, than the file "desc.csv" is loaded.
        - Otherweise the file "fps.csv" is loaded.

    :param fps_as_minusone_and_one: boolean, should binary fingerprints be transformed into {-1,1}
        fingerprints, i.e., 0 = -1, 1 = 1. (default = False)

    :param verbose: boolean, should the function print the number of loaded molecules? (default = False)

    :param rts_as_rank: boolean, should the retention time converted into a dense rank using
        scipy.stats.rankdata? (default = False)

    :return: (pandas.DataFrame, pandas.DataFrame)-tuple

        1) pandas.DataFrame({"retention times": [...], "inchi": [...], "system": [...]})
        2) pandas.DataFrame({"inchi", [...], "fp_1", [...], "fp_2", [...], ..., "fp_N", [...]})

        NOTE: Retention times and features are sorted with respect to their corresponding InChI.
    """
    assert isinstance (predictor, list)

    # Predefined set of molecular descriptors
    l_desc = ["ALOGPDescriptor.ALogP", "ALOGPDescriptor.ALogp2", "ALOGPDescriptor.AMR", "APolDescriptor.apol",
              "AutocorrelationDescriptorPolarizability.ATSp1", "AutocorrelationDescriptorPolarizability.ATSp2",
              "AutocorrelationDescriptorPolarizability.ATSp3", "AutocorrelationDescriptorPolarizability.ATSp4",
              "AutocorrelationDescriptorPolarizability.ATSp5", "BPolDescriptor.bpol",
              "EccentricConnectivityIndexDescriptor.ECCEN", "FractionalPSADescriptor.tpsaEfficiency",
              "KappaShapeIndicesDescriptor.Kier1", "KappaShapeIndicesDescriptor.Kier2",
              "KappaShapeIndicesDescriptor.Kier3", "MannholdLogPDescriptor.MLogP", "VABCDescriptor.VABC",
              "TPSADescriptor.TopoPSA", "WeightDescriptor.MW", "XLogPDescriptor.XLogP"]

    if predictor[0] == "desc" or any ([pred in l_desc for pred in predictor]):
        pred_type = "desc"
    else:
        pred_type = "fps"

    if pred_fn is None:
        pred_fn = "desc.csv" if pred_type == "desc" else "fps.csv"

    # Load retention times
    rts = DataFrame.from_csv (input_dir + "/rts.csv", header = 0, index_col = "inchi")

    # Load molecular representation
    input_file = input_dir + "/" + pred_fn
    data = DataFrame.from_csv (input_file, header = 0, index_col = "inchi")

    # Extract entries of the desired system
    if not (system is None):
        rts_system = rts.loc[rts.system.isin ([system]), :].copy()
    else:
        rts_system = rts.copy()
    data_system = data.loc[rts_system.index].copy()

    if (verbose):
        print ("%s: Number of molecules: %d" % (system, rts_system.shape[0]))

    if rts_as_rank:
        rts_system.rt = sp.stats.rankdata (rts_system.rt, method = "dense")

    # Pre-process fingerprints
    #   - Extract desired definitions
    #   - Encode the as [-1,1]-vectors (optional)
    if pred_type == "desc":
        if predictor[0] != "desc":
            data_system = data_system.filter (regex = "|".join (predictor))
    else:
        if predictor[0] != "fps":
            data_system = data_system.filter (regex = "|".join (predictor))

        if fps_as_minusone_and_one:
            data_system.replace (0, -1, inplace = True)

    # Order the entries in the table in the same way
    rts_system.reset_index (drop = False, inplace = True)
    data_system.reset_index (drop = False, inplace = True)

    rts_system.sort_values ("inchi", inplace = True)
    data_system.sort_values ("inchi", inplace = True)

    assert (rts_system.inchi == data_system.inchi).all()

    return rts_system, data_system

def get_pairwise_labels (pairs, balance_classes = True, random_state = None):
    """
    Task: Get the labels for each pairwise relation. By default the pairs are randomly distributed across
          a positive and negative class:
          y =  1 (positive) : pair = (i,j) with i elutes before j
          y = -1 (negative) : pair = (j,i) with i elutes before j

    :param pairs: list of index pairs considered for the pairwise features
        [(i,j), ...] for which it holds: i elutes before j

    :param balance_classes:  binary indicating, whether the half of the pairs
        should be swapped and assigned a negative target value. This is use-
        full if a binary SVM is training. If a one class SVM is trained,
        than the classes do not need to be balanced.

    :param random_state:

    :return: tuple (pairs_out, y_out):
        pairs_out: List of pairwise relations of length k
        y_out: Target vector, numpy-array, shape (k x 1)
    """
    # Store old random state
    rs_old = np.random.get_state()

    # Set random seed
    np.random.seed (random_state)

    n_pairs = len (pairs)

    pairs_out = pairs.copy()
    y_out = np.ones (n_pairs)

    if balance_classes:
        # Split the example into positive and negative class: 50/50
        idc = np.arange (0, n_pairs)
        np.random.shuffle (idc)
        idc_n = idc[int((np.floor (n_pairs / 2.0))):]

        # Swap label and pair for negative class
        for idx_n in idc_n:
            pairs_out[idx_n] = (pairs_out[idx_n][1], pairs_out[idx_n][0])
            y_out[idx_n] = -1

    # Restore old random stat
    np.random.set_state (rs_old)

    return (pairs_out, y_out)


class KernelRankSVC (BaseEstimator, ClassifierMixin):
    """
    Implementation of the kernelized Ranking Support Vector Classifier.

    The optimization is performed in the dual-space using the condidation gradient (a.k.a. Frank-Wolfe
    algorithm[1,2]). See also the paper for details on the optimization problem.

    [1] An algorithm for quadratic programming, Frank M. and Wolfe P, Navel Research Logistic Quarterly banner,
        1956
    [2] Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization, Jaggi M., Proceedings of the 30th
        International Conference on Machine Learning, 2013

    :param C: scalar, regularization parameter of the SVM (default = 1.0)

    :param kernel: string or callable, determing the kernel used for the data. Logic of this function
        (default = "precomputed")
        - "precomputed": The kernel matrix is precomputed and provided to the fit and prediction function.
        - ["rbf", "polynomial", "linear"]: The kernel is computed by the scikit-learn build it functions
        - callable: The kernel is computed using the provided callable function, e.g., to get the tanimoto
            kernel.

    :param feature_type: string, determing the used pairwise feature (default = "difference")
        - "difference": Feature difference is used: phi_j - phi_i

    :param tol: scalar, tolerance of the change of the alpha vector. (default = 0.001)
        E.g. if "convergence_criteria" == "alpha_change_max" than
         |max (alpha_old - alpha_new)| < tol ==> convergence.

    :param max_iter: scalar, maximum number of iterations (default = 1000)

    :param t_0: scalar, initial step-size (default = 0.1)

    :param step_size_algorithm: string, which step-size calculation method should be used.
        (default = "diminishing")
        - "diminishing": iterative decreasing stepsize
        - "diminishing_2": iterative decreasing stepsize, another formula
        - "fixed": fixed step size t_0
        - "linesearch": step size determined by line-search

        NOTE: Check corresponding functions "_get_step_size_*" for implementation.

    :param gamma: scalar, scaling factor of the gaussian and polynomial kernel. If None, than it will
        be set to 1 / #features.

    :param coef0: scalar, parameter for the polynomial kernel

    :param degree: scalar, degree of the polynomial kernel

    :param kernel_params: dictionary, parameters that are passed to the kernel function. Can be used to
        input own kernels.

    :param slack_type: string, what the slack variables are representing (default = "on_pairs")
        - "on_pairs": there is a slack-variable for each training pair

    :param convergence_criteria: string, how the convergence of the gradient descent should be determined.
        (default = "alpha_change_max")
        - "alpha_change_max": maximum change of the dual variable
        - "gs_change": change of the dual objective
        - "alpha_change_norm": change of the norm of the dual variables

    :param verbose: boolean, should the optimization be verbose (default = False)

    :param debug: scalar, debug level, e.g., calculation duality gap. This increases the complexity.
        (default = 0)

    :param random_state: integer, used as seed for the random generator. The randomness would
        effect the labels of the training pairs. Check the 'fit' and 'get_pairwise_labels' functions
        for details.

    Kernels:
    --------
    "linear": K(X, Y) = <X, Y>
    "polynomial": K(X, Y) = (gamma <X, Y> + coef0)^degree
    "rbf": K(x, y) = exp(- gamma ||x - y||^2)

    SOURCE: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise
    """
    def __init__ (self, C = 1.0, kernel = "precomputed", feature_type = "difference",
                  tol = 0.001, max_iter = 1000, t_0 = 0.1, step_size_algorithm = "diminishing",
                  gamma = None, coef0 = 1, degree = 3, kernel_params = None, slack_type = "on_pairs",
                  convergence_criteria = "alpha_change_max", verbose = False, debug = 0, random_state = None):

        if feature_type not in ["difference"]:
            raise ValueError ("Invalid feature type: %s" % feature_type)

        if slack_type not in ["on_pairs", "on_examples", "on_examples_out"]:
            raise ValueError ("Invalid slack type: %s" % slack_type)

        if convergence_criteria not in ["gs_change", "alpha_change_max", "alpha_change_norm"]:
            raise ValueError ("Invalid convergence criteria: %s" % convergence_criteria)

        # Parameter for the optimization
        self.tol = tol
        self.max_iter = max_iter
        self.step_size_algorithm = step_size_algorithm
        self.convergence_criteria = convergence_criteria

        if self.step_size_algorithm == "diminishing_2" and t_0 != 0.5:
            self.t_0 = 0.5
            warnings.warn ("Stepsize algorithm '%s' requires t_0 = %f. "
                           "The initital stepsize t_0 was set to that value."
                           % (self.step_size_algorithm, self.t_0))
        else:
            self.t_0 = t_0

        # Kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params

        # General Ranking SVM parameter
        self.C = C
        self.slack_type = slack_type
        if self.slack_type != "on_pairs":
            raise ValueError ("Currently slacks can only be assigned to the pairs.")

        # Pair-params and -feature
        self.feature_type = feature_type
        if self.feature_type != "difference":
            raise ValueError ("Currently only difference features are supported.")

        # Debug parameters
        self.verbose = verbose
        self.debug = debug
        self.random_state = random_state

    def fit (self, mol_ids = None, y = None, **fit_params):
        """
        Estimating the parameters of the dual ranking svm with scaled margin.
        The conditional gradient descent algorithm is used to find the optimal
        alpha vector.

        :param mol_ids: array, CURRENTLY NOT USED

        :param y: array-like, shape = (p, 1). Label for each pairwise relation:
            Positive and negative class:
                y =  1 (positive) : pair = (i,j) with i elutes before j
                y = -1 (negative) : pair = (j,i) with i elutes before j

                NOTE: The pairs for the negative class must be swapped.

            The default value is None, which means that in the fitting function the
            pairwise relations are automatically distributed across the positive and
            negative class (50/50 split). See also 'get_pairwise_labels'.

        :param fit_params: dict of string -> object. This dictionary must contain
            the description of the molecules, e.g. as feature vectors or as kernel matrix.

            Molecule description
            {"FX": array-like, shape = (l, d)} if not self.kernel == "precomputed"
            -- or --
            {"KX": array-like, shape = (l, l)} if     self.kernel == "precomputed"

            Pairwise relations
            {"pairs":  List of tuples (i,j) of length p containing the pairwise relations:
            i elutes before j, ... --> [(i,j), (u,v), ...]}

            Pairwise confidence score (optional)
            {"r": array-like, shape = (p, 1)}
            If not set --> assume all pairwise confidences r_ij = 1.

            Initial alpha (optional)
            {"alpha_init": array-like, shape = (p, 1)}
            Alpha must be out of the feasible set: 0 <= alpha_ij <= C for all (i,j) in P.

        """

        fit_params = fit_params["fit_params"]

        if not "pairs" in fit_params.keys():
            raise ValueError ("Pairwise relations must be provided to the fitting methods. "
                              "'pairs' must be added to the 'fit_param' argument (dict)")
        self._pairs_fit = fit_params["pairs"]

        if y is None:
            pairs, y = get_pairwise_labels (self._pairs_fit, random_state = self.random_state)
        else:
            if y.shape[0] != len (self._pairs_fit):
                raise ValueError ("The label vector y and the list of pairwise relations "
                                  "must have the same length: %d vs %d" % (y.shape[0], len (self._pairs_fit)))
            pairs = self._pairs_fit

        p = len (pairs)

        # Handle input data
        if self.kernel == "precomputed":
            if not "KX" in fit_params.keys():
                raise ValueError ("Kernel is 'precomputed', but no kernel matrix is given. "
                                  "'KX' must be added to the 'fit_param' argument (dict).")

            if not fit_params["KX"].shape[0] == fit_params["KX"].shape[1]:
                raise ValueError ("Precomputed kernel matrix must be squared: KX.shape = (%d, %d)."
                                  % fit_params["KX"].shape)

            self._KX_fit = fit_params["KX"]
        else:
            if not "FX" in fit_params.keys():
                raise ValueError ("Kernel is not 'precomputed', but not molecular representation is given. "
                                  "'FX' must be added to the 'fit_param' argument (dict).")

            self._X_fit = fit_params["FX"]
            self._KX_fit = self._get_kernel (self._X_fit)

        # Pairwise confidence
        if "r" in fit_params.keys():
            if not fit_params["r"].shape[0] == p:
                raise ValueError ("Vector of pairwise confidences 'r' must have the same "
                                  "length as the list of pairwise relations: %d vs %d."
                                  % (fit_params["r"].shape[0], p))

            r = fit_params["r"]
        else:
            r = np.ones ((p, 1))


        ## == Conditional gradient algorithm ==
        if self.feature_type == "difference":
            self.A = self._build_A_matrix (pairs, y)
            assert (self.A.shape[1] == self._KX_fit.shape[0])

        # Initialize the alpha-vector (dual variables)
        if "alpha_init" in fit_params.keys():
            if not self._is_feasible (fit_params["alpha_init"], p):
                ValueError ("The provided initial alpha is not in the feasible set. "
                            "Check that 0 <= alpha_ij <= C for all (i,j) in P.")

            alpha = fit_params["alpha_init"]
        else:
            alpha = np.zeros ((p, 1))

        # For debugging purposes, we might want to set the initial iteration.
        if "k_init" in fit_params.keys():
            k_init = fit_params["k_init"]
        else:
            k_init = 1 # Iterations counter

        k = k_init # Iterations counter
        first_iteration = True

        # Track the change of the objective. Compare also the convergence_criteria
        obj_change = np.inf
        self._obj_has_converged = False

        if self.verbose:
            starttime = process_time()
        if self.debug:
            self.f_0s = []
            self.dgs = []
            self.gs = []
            self.rdgs = []

        if self.convergence_criteria == "gs_change":
            obj_change_label = "Dual obj. rel. change"
            self.gs_conv = deque (maxlen = 2)
            self._last_AKAt_y = self._x_AKAt_y (y = alpha)
            self.gs_conv.append (self._evaluate_dual_objective (alpha, r))
        elif self.convergence_criteria == "alpha_change_max":
            obj_change_label = "Alpha change (max)"
        elif self.convergence_criteria == "alpha_change_norm":
            obj_change_label = "Alpha change (norm)"
        else:
            raise ValueError ("Invalid convergence criteria: %s." % self.convergence_criteria)

        while k <= self.max_iter:
            # Determine a feasible update direction
            alpha_delta = self._get_optimal_alpha_bar (alpha, r) - alpha

            if self.debug > 0:
                self.f_0s.append (self._evaluate_primal_objective (alpha, r))
                self.gs.append (self._evaluate_dual_objective (alpha, r))
                self.dgs.append (self.f_0s[-1] - self.gs[-1])
                self.rdgs.append (self.tol / 2 * (np.abs (self.f_0s[-1]) + np.abs (-self.gs[-1])))

            # Determine the stepwith
            if self.step_size_algorithm == "diminishing":
                t = self._get_step_size_diminishing (k)
            elif self.step_size_algorithm == "linesearch":
                t = self._get_step_size_linesearch (alpha, alpha_delta, r)
            elif self.step_size_algorithm == "fixed":
                t = self.t_0
            elif self.step_size_algorithm == "diminishing_2":
                if first_iteration:
                    t = self.t_0
                else:
                    t = self._get_step_size_diminishing_2 (t)
            else:
                raise ValueError ("Invalid step size algorithm: %s." % self.step_size_algorithm)

            if t <= 0:
                break

            # Store currently best alpha and update
            alpha_old = alpha
            alpha = alpha + t * alpha_delta # alpha_t+1 = alpha_t + tau * (alpha* - alpha_t)
                                            #           = (1 - tau) * alpha_t + tau * alpha*

            if not self._is_feasible (alpha, p):
                self.error_state = {"alpha_old": alpha_old, "alpha_new": alpha, "alpha_delta": alpha_delta, "t": t}
                raise RuntimeError ("Alpha is after update not in the feasible set anymore.")

            if self.convergence_criteria == "gs_change":
                # Calculate the relative change of the dual objective
                self.gs_conv.append (self._evaluate_dual_objective (alpha, r))

                gs_old = self.gs_conv[-2] # alpha_old
                gs     = self.gs_conv[-1] # alpha

                obj_change = np.abs (gs / gs_old - 1) # (gs - gs_old) / gs_old
            elif self.convergence_criteria == "alpha_change_max":
                # Calculate the maximum change of the updated alpha
                obj_change = np.max (np.abs (alpha - alpha_old))
            elif self.convergence_criteria == "alpha_change_norm":
                # Calculate the norm of the change of the updated alpha
                obj_change = np.linalg.norm (alpha - alpha_old)
            else:
                raise ValueError ("Invalid convergence criteria: %s." % self.convergence_criteria)

            if self.verbose and  k % 50 == 0:
                print ("\rIteration %d: %s = %f, Step size t = %f" % (k, obj_change_label, obj_change, t),
                       end = "", flush = True)
                sleep (0.25)

            if (k >= 5) and obj_change < self.tol:
                self._obj_has_converged = True
                break
            else:
                self._obj_has_converged = False
                k += 1

            first_iteration = False

        self._k_convergence = k if self._obj_has_converged else (k - 1)
        if self.verbose:
            print ("\r", end = "", flush = True)
            print ("\rIteration %d: %s = %f, Step size t = %f" % (self._k_convergence, obj_change_label, obj_change, t))
            print ("Convergence: Time = %.3fs" % (process_time() - starttime))

        if self._k_convergence == self.max_iter:
            warnings.warn ("Optimization algorithm stopped due to maximum number of iterations.")
        self._t_convergence = t

        # Find the indices of the support vectors
        self.alpha = alpha.flatten()
        self.idx_sv = (self.alpha > 0).flatten()
        if self.verbose:
            print ("Number of support vectors: %d (out of %d)." % (np.sum (self.idx_sv), self.alpha.shape[0]))

    def _get_kernel (self, X, Y = None, n_jobs = 1):
        """

        :param X: array-like, shape = (n_samples_a, n_features)

        :param Y: array-like, shape = (n_samples_b, n_features), (default = None)

        :param n_jobs: integer, number of jobs passed to 'pairwise_kernels' (default = 1)

        :return: array-like, Kernel matrix between feature sets A and A or A and B, shape:
            (n_samples_a, n_samples_a) if     Y is None
            (n_samples_a, n_samples_b) if not Y is None
        """
        if callable (self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels (X, Y, metric = self.kernel, filter_params = True, n_jobs = n_jobs, **params)

    def predict (self, X1, X2 = None, n_jobs = 1):
        """
        Predict for each example * in {1,...,n} (set 1) whether it elutes before or after the
        example u (for each u in {1,...,m}) (set 2)

        For example:
            set 1: test set
            set 2: training set

        :param X1: array-like, data description
            feature-vectors: shape = (n, d)
            -- or --
            kernel-matrix: shape = (l, n),

            with n being the number of molecules in set 1, e.g. the test set.

        :param X2: array-like, data description (default: None)
            None: --> use the training set (with m = l examples)
            -- or --
            feature-vectors: shape = (m, d)
            -- or --
            kernel-matrix: shape = (l, m)

            with m being the number of molecules in set 2, e.g. the training set.

        :param n_jobs: integer, number of jobs passed to '_get_kernel' (default = 1)

        :return: array-like, shape = (n, m). Entry at index [*, u] is either:
             1: * (set 1 example) elutes before u (set 2 example)
            -1: u (set 2 example) elutes before * (set 1 example)

            ==> For each test example there will be a prediction which of the training molecules
                elute before respectively after the test example.
        """

        if self.kernel == "precomputed":
            if not X1.shape[0] == self._KX_fit.shape[0]:
                raise ValueError ("Set 1 kernel must have as many rows as training examples.")
        else:
            X1 = self._get_kernel (X1, self._X_fit, n_jobs = n_jobs).T

        if X2 is None:
            X2 = self._KX_fit
        else:
            if not self.kernel == "precomputed":
                X2 = self._get_kernel (X2, self._X_fit, n_jobs = n_jobs).T

            if not X2.shape[0] == self._KX_fit.shape[0]:
                raise ValueError ("Set 2 kernel must have as many rows as training examples.")

        n = X1.shape[1]
        m = X2.shape[1]
        Y = np.zeros ((n, m))

        if self.feature_type == "difference":
            B = (self.A.T.dot (self.alpha))
            for idx in range (n):
                f2_minus_f1 = (X2.T - X1.T[idx]).dot (B)
                f2_minus_f1[np.isclose (f2_minus_f1, 0, atol = 1e-10)] = 0
                Y[idx, :] = np.sign (f2_minus_f1)

        return (Y)

    def _map_values (self, X, alpha, idx_sv = None):
        if idx_sv is None:
            idx_sv = (alpha > 0).flatten()

        if self.feature_type == "difference":
            Y = (alpha * self.A).dot (X)
        else:
            raise NotImplemented ("Currently only difference features are implemented.")

        return (Y)


    def map_values (self, X, n_jobs = 1):
        """
        Task: Calcualte w^T\phi for a set of examples.

        :param X: array-like, data description
            feature-vectors: shape = (n, d)
            -- or --
            kernel-matrix: shape = (l, n),

            with n being the number of molecules in set 1, e.g. the test set.

        :param n_jobs: integer, number of jobs passed to '_get_kernel' (default = 1)

        :return: array-like, shape = (n, ), mapped values for all examples.
        """
        if self.kernel == "precomputed":
            if not X.shape[0] == self._KX_fit.shape[0]:
                raise ValueError ("Train-test kernel must have as many rows as training examples.")
        else:
            X = self._get_kernel (X, self._X_fit, n_jobs = n_jobs).T # shape = (l, n)

        return (self._map_values (X, self.alpha, self.idx_sv))

    def score (self, X, pairs, sample_weight = None, y = None, normalize = True):
        """
        :param X: array-like, data description
            feature-vectors: shape = (n, d)
            -- or --
            kernel-matrix: shape = (l, n),

            with n being the number of test molecules.

        :param pairs: List of tuples (i,j) of length p containing the pairwise relations:
            i elutes before j, ... --> [(i,j), (u,v), ...]

        :param sample_weight: parameter is ignored at the moment!

        :return: scalar, mean pairwise prediction accuracy
        """
        if len (pairs) == 0:
            return (0.0)

        if y is None:
            _, y = get_pairwise_labels (pairs, balance_classes = False)

        Y = self.predict (X, X)

        tp = 0.0
        for idx, pair in enumerate (pairs):
            if Y[pair[0], pair[1]] == 0 and pair[0] != pair[1]:
                tp += 0.5
            else:
                tp += (Y[pair[0], pair[1]] == y[idx])

            assert (Y[pair[0], pair[1]] == 0 or Y[pair[0], pair[1]] == -Y[pair[1], pair[0]])

        if normalize:
            return tp / len (pairs)
        else:
            return tp

    def score_using_prediction (self, Y, pairs, normalize = True):
        if len (pairs) == 0:
            return (0.0)

        _, y = get_pairwise_labels (pairs, balance_classes = False)

        tp = 0.0
        for idx, pair in enumerate (pairs):
            if Y[pair[0], pair[1]] == 0 and pair[0] != pair[1]:
                tp += 0.5
            else:
                tp += (Y[pair[0], pair[1]] == y[idx])

            assert (Y[pair[0], pair[1]] == 0 or Y[pair[0], pair[1]] == -Y[pair[1], pair[0]])

        if normalize:
            return tp / len (pairs)
        else:
            return tp

    def _scorer (self, X, pairs, y):
        return self.score (X, pairs = pairs, y = y)

    def _evaluate_primal_objective (self, alpha, r):
        """
        Get the function value of f_0:
            f_0(w(alpha), xi(alpha, r)) = 0.5 w(alpha)^Tw(alpha) + C 1^Txi(alpha, r)

        :param alpha: array-like, shape = (p, 1), current alpha estimate.
        :param r: array-like, shape = (p, 1), confidence for each pairwise relation.

        :return: scalar, f_0(w(alpha), xi(alpha, r))
        """
        if self.feature_type == "difference":
            wtw = alpha.T.dot (self._last_AKAt_y)
        else:
            raise ValueError ("Invalid feature type: %s." % self.feature_type)

        if self.slack_type == "on_pairs":
            # See Juho's lecture 4 slides: "Convex optimization methods", slide 38
            xi = np.maximum (0, r - self._last_AKAt_y)
        else:
            raise ValueError ("Invalid slack type: %s." % self.slack_type)

        return (0.5 * wtw + self.C * np.sum (xi))[0]

    def _evaluate_dual_objective (self, alpha, r):
        """
        Get the function of g:
            g(alpha, r) = r^T alpha - 0.5 alpha^T H alpha

        :param alpha: array-like, shape = (p, 1), current alpha estimate.
        :param r: array-like, shape = (p, 1), confidence for each pairwise relation.
        :return: scalar, g(alpha, r)
        """
        if self.feature_type == "difference":
            return (r.T.dot (alpha) - 0.5 * alpha.T.dot (self._last_AKAt_y))[0]

    def _is_feasible (self, alpha, p):
        """
        Check whether the provided alpha vector is in the feasible set:

        For all slack-types:
            1) 0 <= alpha_ij for all (i,j) in P
            2) alpha.shape[0] == p

        If slack-type: on_pairs:
            3) alpha_ij <= C for all (i,j) in P

        :param alpha: array-like, shape = (p, 1), dual variables

        :param p: scalar, number of dual variables (pairwise relations)

        :return: Boolean:
            True,  if alpha is in the feasible set
            False, otherwise
        """
        is_feasible = (0 <= alpha).all() and alpha.shape[0] == p

        if is_feasible:
            if self.slack_type == "on_pairs":
                is_feasible = is_feasible and (alpha <= self.C).all()
            else:
                raise ValueError ("Invalid slack type: %s" % self.slack_type)

        return (is_feasible)

    def _get_optimal_alpha_bar (self, alpha, r):
        """
        Finding the alpha*

        :param r: array-like, shape = (p, 1), confidence values for the pairwise
            relations

        :return: array-like, shape = (p, 1), alpha*
        """
        if self.feature_type == "difference":
            self._last_AKAt_y = self._x_AKAt_y (y = alpha)
            d = r - self._last_AKAt_y
        else:
            raise ValueError ("Invalid feature type: %s." % self.feature_type)

        if self.slack_type == "on_pairs":
            alpha_bar = np.zeros ((self.A.shape[0], 1))
            alpha_bar[d > 0] = self.C
        else:
            raise ValueError ("Invalid slack type: %s." % self.slack_type)

        return alpha_bar

    def _x_AKAt_y (self, x = None, y = None):
        """
        Function calculating:
            1) x^T \widetilde{A} K_phi \widetilde{A}^T y, if x and y are not None
            2) \widetilde{A} K_phi \widetilde{A}^T y,     if x is None and y is not None
            3) x^T\widetilde{A} K_phi \widetilde{A}^T,    if x is not None and y is None
            4) \widetilde{A} K_phi \widetilde{A}^T,       if x and y are None

        :param x: array-like, shape = (p, 1), vector on the left side

        :param y: array-like, shape = (p, 1), vector on the right side

        :return: array-like (dense),
            1) shape = (1, 1), if x and y are not None
            2) shape = (p, 1), if x is None and y is not None
            3) shape = (1, p), if x is not None and y is None
            4) shape = (p, p), if x and y are None
        """
        if (x is not None) and (y is not None):
            return (x.T.dot(
                self.A.dot (
                    (self._KX_fit.dot (
                        (self.A.T.dot (y)) # 1.
                    ))                     # 2.
                )                          # 3.
            ))                             # 4.
        elif (x is None) and (y is not None):
            return (self.A.dot (
                (self._KX_fit.dot (
                    (self.A.T.dot (y)) # 1.
                ))                     # 2.
            ))                         # 3.
        elif (x is not None) and (y is None):
            return (self.A.dot((((self.A.T.dot(x)).T).dot (self._KX_fit)).T).T)
        else:
            return (self.A.dot (self._KX_fit).dot (self.A.T))

    def _build_A_matrix (self, pairs, y = None):
        """
        Construct a matrix A (p x l) so that:
            A_{(i,j),:} = y_ij (0...0, -1, 0...0, 1, 0...0).

        This matrix is used to simplify the optimization using 'difference' features.

        :param pairs: List of tuples (i,j) of length p containing the pairwise relations:
            i elutes before j, ... --> [(i,j), (u,v), ...]

        :param y: array-like, shape = (p, 1). Label for each pairwise relation (optional):
            Positive and negative class:
                y =  1 (positive) : pair = (i,j) with i elutes before j
                y = -1 (negative) : pair = (j,i) with i elutes before j

            If y is None: it is assumed that all pairs belong to the positive class.

        :return: sparse array-like, shape = (p, l)
        """
        p = len (pairs)
        l = self._KX_fit.shape[0]

        row_ind = np.append (np.arange (0, p), np.arange (0, p))
        col_ind = np.append ([pair[0] for pair in pairs], [pair[1] for pair in pairs])
        data = np.append (-1 * np.ones ((1, p)), np.ones ((1, p)))

        if not y is None:
            data = data * np.append (y, y)

        return (sp.sparse.csr_matrix ((data, (row_ind, col_ind)), shape = (p, l)))

    def _get_step_size_diminishing (self, k):
        """
        Calculate the step size using the diminishing strategy.

        :param k: scalar, current iteration
        :return: scalar, step size
        """
        assert k > 0

        return self.t_0 / (1 + self.t_0 * self.C * (k - 1))

    def _get_step_size_diminishing_2 (self, t):
        """
        Step size after Sandor:

            t = t - (t**2 / 2)

        :param t: scalar, current step size
        :return: scalar, step size
        """
        return t - (t**2 / 2.0)

    def _get_step_size_linesearch (self, alpha, alpha_delta, r):
        """
        Calculate the step size using the linear search algorithm.

        :param alpha:
        :param alpha_delta:
        :param r:
        :return:
        """

        if self.feature_type == "difference":
            x_AKAt = self._x_AKAt_y (x = alpha_delta)
            return ((alpha_delta.T.dot(r) - x_AKAt.dot (alpha)) / x_AKAt.dot (alpha_delta))