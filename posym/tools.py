from itertools import combinations
import numpy as np


def unique_rep(rep):
    unique_list = []
    ref_list = []

    for r in rep:
        if tuple(r) not in ref_list:
            unique_list.append(r)
            ref_list.append(tuple(r))

    return unique_list


def get_reps(r_original, tables):
    for ai in range(1, len(tables)):
        for r_list in combinations(tables.iteritems(), ai):
            r_sum = r_list[0][1] * 0
            for ri in r_list:
                r_sum += ri[1]

            if np.all(r_sum.sort_index() == r_original.sort_index()):
                return [ri[1] for ri in r_list]
