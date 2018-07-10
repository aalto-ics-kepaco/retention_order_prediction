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

import time
import numpy
import itertools

from collections import OrderedDict

def get_statistic_about_concordant_and_discordant_pairs(pairs, keys, perform_checks=True):
    """
    Count the number of pair occurrences using the molecular ids instead of
    the row indices. This can be used to for example determine the pairs that
    are discordant across different systems.

    :param pairs: list of tuples, shape = (p,), list of pairs given as
                  tuples:

                    (i,j) --> m_i elutes before m_j.

                  The indices i and j are given as positive integers.

    :param keys: list of tuples, shape = (p,), list of (mol-id, system)
                 tuples. The indices of the pairs are corresponding to
                 indices in the key list:

                    (i,j) --> keys[i][0] elutes before keys[j][0],
                        in system keys[i][1], with keys[i][1] == keys[j][1]

    :param perform_checks: boolean, should consitency checks be performed. This
                           increases the computational complexity. (default = True)

    :return: dictonary:

             keys: (m_i,m_j)-tuples

             values: dictionary = {
                "#Pij>": Number of m_i elutes before m_j occurrences,
                "#Pij<": Number of m_j elutes before m_i occurrences}

             NOTE: For the keys, the first (m_i,m_j) occurrences is takes as
                   "reference". If the second pair would be (m_j,m_i), than
                   just the "#Pij<" counter would be increased, i.e. that for
                   each pair (regardless of its order) only one element is in
                   the dictionary.
    """
    if len (pairs) == 0:
        return {}

    if not len (keys):
        raise ValueError ("If pairs are provided, than the key-list must not be empty.")

    d_pairs_stats = {}
    for i, j in pairs:
        m_i, m_j = keys[i][0], keys[j][0]

        if (m_i, m_j) not in d_pairs_stats.keys() and (m_j, m_i) not in d_pairs_stats.keys():
            d_pairs_stats[(m_i, m_j)] = {"#Pij>": 1, "#Pij<": 0, "Pij": {(i,j)}}
        elif (m_i, m_j) in d_pairs_stats.keys():
            d_pairs_stats[(m_i, m_j)]["#Pij>"] += 1
            d_pairs_stats[(m_i, m_j)]["Pij"] |= {(i,j)}
        elif (m_j, m_i) in d_pairs_stats.keys():
            d_pairs_stats[(m_j, m_i)]["#Pij<"] += 1
            d_pairs_stats[(m_j, m_i)]["Pij"] |= {(i,j)}

    # Make some consistency checks
    if perform_checks:
        assert (len(d_pairs_stats) <= len(pairs))
        n_systems = len(numpy.unique(list(zip(*keys))[1]))
        n_pairs_out = 0
        for stats in d_pairs_stats.values():
            assert(stats["#Pij<"] <= n_systems)
            assert(stats["#Pij>"] <= n_systems)
            n_pairs_out += len (stats["Pij"])
        assert (n_pairs_out == len (pairs))

    return d_pairs_stats

def _sample_perc_from_list(lst, perc=100, algorithm="cum_rand", random_state=None):
    """
    Sample randomly a certain percentage of items from the given
    list. The original order of the items is kept.

    :param lst: list, shape = (n,), input items

    :param perc: scalar, percentage to sample

    :param algorithm: string, which algorithm should be used

        "random": Decide for each item to be chosen or not. This
                  algorithm runs in linear time O(n), but
                  the percentages might not match exactly.

        "cum_rand": O(n log(n) + perc)

    :return: list
    """
    if perc >= 100:
        return lst
    if perc <= 0:
        return []

    # Store old random state and set random state
    rs_old = numpy.random.get_state()
    numpy.random.seed(random_state)

    if algorithm == "random":
        lst_sub = [it for it in lst if numpy.random.uniform(high=100) <= perc]
    elif algorithm == "cum_rand":
        n = len(lst)
        n_perc = numpy.round(n * perc / 100.0)
        rank_its = numpy.argsort(numpy.random.uniform(size=n))
        lst_sub = []
        for idx, it in enumerate(lst):
            if rank_its[idx] < n_perc:
                lst_sub.append(it)

            if len(lst_sub) > n_perc:
                break
    else:
        raise ValueError("Invalid sampling algorithm: %s." % algorithm)

    # Restore old random stat
    numpy.random.set_state(rs_old)

    return lst_sub

def pairwise(iterable):
    """
    source: https://docs.python.org/3/library/itertools.html#itertools.combinations

    :param iterable: s
    :return: s -> (s0, s1), (s1, s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def is_sorted (l, ascending = True):
    """
    Check whether array is sorted.

    source: https://stackoverflow.com/questions/3755136/pythonic-way-to-check-if-a-list-is-sorted-or-not

    :param l: list
    :return: is sorted
    """
    if ascending:
        return all (l[i] <= l[i+1] for i in range (len (l)-1))
    else:
        return all (l[i] >= l[i+1] for i in range (len (l)-1))


def sample_perc_from_list(lst, tsystem=None, perc=100, algorithm="cum_rand", random_state=None):
    """
    Sample randomly a certain percentage of items from the given
    list. The original order of the items is kept.

    :param lst: list of tuples, shape = (n,), input items (mol_id, system_id)

    :param perc: scalar, percentage of examples to sample from the list

    :param tsystem: string, system_id to consider for the sampling (default = None)
        None: Sample simply from the list, without considering the system.
        systen_id: Apply sampling only for the specified system. All other
            list-elements are simply copied.

    :param perc: scalar, percentage to sample

    :param algorithm: string, which algorithm should be used

        "random": Decide for each item to be chosen or not. This
                  algorithm runs in linear time O(n), but
                  the percentages might not match exactly.

        "cum_rand": O(n log(n) + perc)

    :return: list
    """
    if tsystem is None:
        return _sample_perc_from_list(lst, perc=perc, algorithm=algorithm, random_state=random_state)
    if perc >= 100:
        return lst

    lst_of_systems = list(zip(*lst))[1]
    if tsystem not in lst_of_systems:
        return lst

    # Store old random state and set random state
    rs_old = numpy.random.get_state()
    numpy.random.seed(random_state)

    if algorithm == "random":
        lst_sub = [it for it in lst if it[1] != tsystem or numpy.random.uniform(high=100) <= perc]
    elif algorithm == "cum_rand":
        n_tsys = numpy.sum([1 for it in lst if it[1] == tsystem])  # O(n)
        n_tsys_perc = numpy.round(n_tsys * perc / 100.0)
        rank_tsys_its = numpy.argsort(numpy.random.uniform(size=n_tsys))  # O(n_tsys + n_tsys log (n_tsys))
        lst_sub = []
        idx = 0
        for it in lst:  # O(n)
            if it[1] != tsystem:
                lst_sub.append(it)
            else:
                if rank_tsys_its[idx] < n_tsys_perc:
                    lst_sub.append(it)

                idx += 1
    else:
        # FIXME: Reset the random state, if an exception ins thrown.
        raise ValueError("Invalid sampling algorithm: %s." % algorithm)

    # Restore old random stat
    numpy.random.set_state(rs_old)

    return lst_sub

def dict2str(d, sep="-", sort_names=True):
    """
    Concatenate key-value pairs to string.

    :param d:
    :param sep: string, separating the names (dict keys) (default = "-")
    :param sort_names: binary, indicating whether the names should be
                       sorted alphabetically             (default = True)
    :return:
    """
    if d is None:
        return None

    ostr = ""
    keys = list(d.keys())
    if sort_names:
        keys = sorted(keys)

    for key in keys:
        if d[key] is None:
            continue

        if ostr == "":
            if str(key) == "":
                ostr = "".join([str(key), str(d[key])])
            else:
                ostr = "=".join([str(key), str(d[key])])
        else:
            if str(key) == "":
                ostr = sep.join([ostr, "".join([str(key), str(d[key])])])
            else:
                ostr = sep.join([ostr, "=".join([str(key), str(d[key])])])

    return ostr


def split_with_minimum_rt_distance(rts, min_rt_delta=0, random_state=None):
    """
    Sample from a set ot retention times, so that the sampled rts have a
    minimum rt differences.

    :param rts:
    :param min_rt_delta:
    :param random_state:
    :return:
    """
    # if min_rt_delta == 0:
    #     return list(range(len(rts)))

    # Store old random state and set random state
    rs_old = numpy.random.get_state()
    numpy.random.seed(random_state)

    last_rt = -numpy.inf
    idc = []
    for rt in numpy.unique(rts):
        if last_rt + min_rt_delta <= rt:
            sel = numpy.where(rts == rt)[0]
            idc.append (sel[numpy.random.randint(0,len(sel))])
            last_rt = rt

    # Restore old random state
    numpy.random.set_state(rs_old)

    return idc


def join_dicts(d, keys=None):
    """
    Task: Concatenate list/directory of dictionaries: [d1, d2, ... ] or {"d1": d1, "d2": d2, ...}
          into a single dictionary.

          Note: This function returns a ordered dictionary, i.e, the order of the key insertions
                is preserved when for example .keys()-function is used.

    :param d: List or directory of directories
    :param keys: Range for lists
                 Keys for dictionaries
    :return: Single ordered dictionary containing all the (key, value)-pairs
             from the separate dictionaries.

    :example:
        {"s1": {("mol1","s1"): [...], ("mol2","s1"): [...], ...},
         "s2": {("mol1","s2"): [...], ("mol3","s2"): [...], ...}}

        -->

        {("mol1","s1"): [...], ("mol2","s1"): [...], ..., ("mol1","s2"): [...], ("mol3","s2"): [...], ...}
    """
    if keys is None:
        keys = d.keys()

    d_out = OrderedDict()
    for key in keys:
        d_out.update(d[key])

    return d_out


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s] Elapsed: %.3f' % (self.name, (time.time() - self.tstart)))
        else:
            print('Elapsed: %.3f' % (time.time() - self.tstart))
