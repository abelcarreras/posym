from posym.tools import unique_rep, get_representation
from posym.pointgroup import PointGroup
import numpy as np
import pandas as pd


class SymmetryBase():
    """
    This class is supposed to be used as a base for more complex symmetry objects

    """
    def __init__(self, group, rep, coefficients=None):

        self._pg = PointGroup(group)
        self._table = self._pg.ir_table

        self._group = group
        self._coeff = {}
        self._rep = []

        if isinstance(rep, str):
            if rep not in self._table:
                raise Exception('Representation do not match with group')
            self._rep.append(self._table[rep])
            self._coeff[tuple(self._table[rep])] = 1

        elif isinstance(rep, pd.Series):
            for r_name, r in self._table.items():
                try:
                    if np.all(r.sort_index() == rep.sort_index()):
                        self._rep.append(r)
                        self._coeff[tuple(self._table[r])] = 1
                        return
                except:
                    raise Exception('Representation do not match with group')
            raise Exception('Representation not in group')

        elif isinstance(rep, list):
            self._coeff = {}
            for k, ri in enumerate(rep):

                for r in get_representation(ri, self._table):
                    if tuple(r) in self._coeff:
                        self._coeff[tuple(r)] += coefficients[tuple(ri)]
                    else:
                        self._coeff[tuple(r)] = coefficients[tuple(ri)]

                self._rep += get_representation(ri, self._table)
                continue

                # just for test
                for r_name, r in self._table.items():
                    try:
                        if np.all(r.sort_index() == ri.sort_index()):
                            self._rep.append(r)
                            self._coeff[tuple(r)] = coefficients[tuple(ri)]
                    except:
                        raise Exception('Representation do not match with group')

        else:
            raise Exception('Representation format not valid')

    def print_representation(self):
        # return '{}\n--------\n{}'.format(self._rep.name, self._rep.to_string())
        str = ''
        for r in self._rep:
            str += r.__str__() + ', coeff: {}'.format(self._coeff[tuple(r)]) + '\n'
        print(str)

    def get_representation(self):

        r_total = self._rep[0] * 0
        for r in self._rep:
            r_total += r * self._coeff[tuple(r)]

        r_total.name = self.__repr__()
        return r_total

    def get_point_group(self):
        return self._pg

    def __str__(self):
        str = ''
        for i, r in enumerate(self._rep):
            if i > 0:
                str += ' + '
            if self._coeff[tuple(r)] > 1:
                str += '{} {}'.format(self._coeff[tuple(r)], r.name)
            else:
                str += r.name

        return str

    def __repr__(self):
        str = ''
        for i, r in enumerate(self._rep):
            if i > 0:
                str += '+'
            if self._coeff[tuple(r)] > 1:
                str += '{}{}'.format(self._coeff[tuple(r)], r.name)
            else:
                str += r.name
        return str

    def __add__(self, other):

        def merge_coeff(x, y):
            z = x.copy()
            for item in y.items():
                if item[0] in z:
                    z[item[0]] += item[1]
                else:
                    z[item[0]] = item[1]
            return z

        if self._group == other._group:
            return SymmetryBase(self._group,
                                unique_rep(self._rep + other._rep),
                                coefficients=merge_coeff(self._coeff, other._coeff))

        raise Exception('Incompatible point groups')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):

        if isinstance(other, int):
            return SymmetryBase(self._group,
                                self._rep,
                                coefficients={r: v*other for r, v in self._coeff.items()})

        elif isinstance(other, SymmetryBase):

            r_list = []
            new_coeff = {}
            for rep in self._rep:
                for rep_other in other._rep:
                    r1 = rep_other.values
                    r2 = rep.values
                    r_index = rep.index

                    rf = pd.Series(np.multiply(r1, r2).tolist(), index=r_index)
                    r_list.append(rf)
                    if tuple(rf) in new_coeff:
                        new_coeff[tuple(rf)] += self._coeff[tuple(rep)] * other._coeff[tuple(rep_other)]
                    else:
                        new_coeff[tuple(rf)] = self._coeff[tuple(rep)] * other._coeff[tuple(rep_other)]

            return SymmetryBase(self._group, unique_rep(r_list), coefficients=new_coeff)
        else:
            raise Exception('Symmetry operation not possible')

