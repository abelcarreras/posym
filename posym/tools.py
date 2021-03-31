from itertools import combinations
import numpy as np


def list_round(list, decimals=2):
    r_list = []
    for element in list:
        if abs(np.round(element) - element) < 10**(-decimals):
            r_list.append(int(np.round(element)))
        else:
            r_list.append(np.round(element, decimals))

    return r_list
