import numpy as np
from posym.permutations import get_cross_distance_table, get_permutation_simple


def get_cross_distance_table_py(coordinates_1, coordinates_2):

    coordinates_1 = np.array(coordinates_1)
    coordinates_2 = np.array(coordinates_2)

    distances = np.zeros((len(coordinates_1), len(coordinates_2)))

    for i, c1 in enumerate(coordinates_1):
        for j, c2 in enumerate(coordinates_2):
            # print(i, j, c1, c2, np.linalg.norm(c1 - c2))
            distances[i, j] = np.linalg.norm(c1 - c2)
    return distances


def get_permutation_simple_py(distance_table, symbols):

    def evaluation(i, j, row1, row2):
        from scipy.stats import norm

        s1 = np.sum([val**2 * norm.pdf(k, loc=i, scale=0.1) for k, val in enumerate(row1)])
        s2 = np.sum([val**2 * norm.pdf(k, loc=j, scale=0.1) for k, val in enumerate(row2)])

        return s1 + s2

    def evaluation_fast(i, j, row1, row2):
        off_diagonal = (row1[j] - row2[i]) ** 2
        return row1[i] ** 2 + row2[j] ** 2 + off_diagonal

    perm = np.array(range(len(distance_table)))
    distance_table = np.array(distance_table).copy()

    control = True
    while control:
        control = False
        for a in range(len(distance_table)-1, 1, -1):
            for i in range(len(distance_table)):
                j = i+a if i+a < len(distance_table) else i+a - len(distance_table)

                row_1 = distance_table[i]
                row_2 = distance_table[j]

                if symbols[i] == symbols[j]:
                    if evaluation_fast(i, j, row_1, row_2) > evaluation_fast(j, i, row_1, row_2):
                        distance_table[[i, j]] = distance_table[[j, i]]
                        perm[[i, j]] = perm[[j, i]]
                        control = True

                # print(i, j, evaluation_fast(i, j, row_1, row_2), evaluation_fast(j, i, row_1, row_2))
                # print(distance_table)

    def inverse_perm(perm):
        return [list(perm).index(j) for j in range(len(perm))]

    # print('-->\n', np.round(distance_table, decimals=3))
    # print(np.round(np.diag(distance_table), decimals=3))
    return inverse_perm(perm)


class Operation:
    def __init__(self, label):
        self._label = label

    def get_permutation(self, operation, coordinates, symbols):
        operated_coor = np.dot(operation, coordinates.T).T

        symbols = [int.from_bytes(num.encode(), 'big') for num in symbols]
        distance_table = get_cross_distance_table(coordinates, operated_coor)
        perm = get_permutation_simple(distance_table, symbols)
        permu_coor = operated_coor[list(perm)]

        measure = np.average(np.linalg.norm(np.subtract(coordinates, permu_coor), axis=1))

        return measure, list(perm)

    def get_normalization(self, coordinates):

        sum_list = []
        for r1 in coordinates:
            for r2 in coordinates:
                subs = np.subtract(r1, r2)
                sum_list.append(np.dot(subs, subs))
        d = np.average(sum_list)

        return d

    @property
    def label(self):
        return self._label
