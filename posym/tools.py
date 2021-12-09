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


def standardize_vector(vector, prec=1e-5):

    vector = np.array(vector, dtype=float)
    if np.abs(vector[0]) > prec:
        if vector[0] < 0:
            vector = np.array(vector) * -1
    elif np.abs(vector[1]) > prec:
        if vector[1] < 0:
            vector = np.array(vector) * -1
    else:
        if vector[2] < 0:
            vector = np.array(vector) * -1

    return vector.tolist()
