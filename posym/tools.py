from itertools import combinations
import numpy as np


def unique_rep(representations_list):
    """
    get unique representation list from list of representations.
    Basically used to simplify representation description in the basis of IR

    :param representations_list: list or representations
    :return: unique list of representations
    """
    unique_list = []
    ref_list = []

    for r in representations_list:
        if tuple(r) not in ref_list:
            unique_list.append(r)
            ref_list.append(tuple(r))

    return unique_list


def get_representation(r_original, ir_table):
    """
    Obtain a standardized representation from IR table of the point group

    :param r_original: not standardized representation
    :param ir_table: IR table
    :return: standardized representation
    """

    for i in range(1, len(ir_table)):
        for r_list in combinations(ir_table.iteritems(), i):
            r_sum = r_list[0][1] * 0
            for ri in r_list:
                r_sum += ri[1]

            if np.all(r_sum.sort_index() == r_original.sort_index()):
                return [ri[1] for ri in r_list]
