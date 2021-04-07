from posym.tools import list_round
from posym.pointgroup import PointGroup
import numpy as np
import pandas as pd


class SymmetryBase():
    """
    This class is supposed to be used as a base for more complex symmetry objects

    """
    def __init__(self, group, rep):

        self._pg = PointGroup(group)
        self._group = group

        ir_table = self._pg.ir_table


        if isinstance(rep, str):
            if rep not in ir_table:
                raise Exception('Representation do not match with group\nAvailable: {}'.format(ir_table.T.index))
            self._op_representation = ir_table[rep]

        elif isinstance(rep, pd.Series):
            if np.all(ir_table.sort_index().index == rep.sort_index().index):
                self._op_representation = rep.reindex(ir_table.index)
            else:
                raise Exception('Representation not in group')

    def get_op_representation(self):
        return self._op_representation

    def get_ir_representation(self):
        ir_rep = np.dot(self._pg.trans_matrix_inv, self._op_representation.values)
        return pd.Series(ir_rep, index=self._pg.ir_table.T.index)

    def get_point_group(self):
        return self._pg

    def __str__(self):

        ir_rep = self.get_ir_representation().values
        ir_rep = list_round(ir_rep)

        ir_labels = self.get_ir_representation().index

        str = ''
        for i, r in enumerate(ir_rep):
            if np.sum(ir_rep[:i]) > 0 and r > 0:
                str += ' + '
            if r == 1:
                str += ir_labels[i]
            elif r > 0:
                str += '{} {}'.format(r, ir_labels[i])

        return str

    def __repr__(self):
        ir_rep = self.get_ir_representation().values
        ir_rep = list_round(ir_rep)

        ir_labels = self.get_ir_representation().index

        str = ''
        for i, r in enumerate(ir_rep):
            if np.sum(ir_rep[:i]) > 0 and r > 0:
                str += '+'
            if r == 1:
                str += ir_labels[i]
            elif r > 0:
                str += '{}{}'.format(r, ir_labels[i])
        return str

    def __add__(self, other):

        if self._group == other._group:
            return SymmetryBase(self._group,
                                self._op_representation + other._op_representation)

        raise Exception('Incompatible point groups')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):

        if isinstance(other, (float, int)):
            return SymmetryBase(self._group,
                                self._op_representation * other)

        elif isinstance(other, SymmetryBase):
            mul_rep = self._op_representation * other._op_representation

            return SymmetryBase(self._group, mul_rep)
        else:
            raise Exception('Symmetry operation not possible')

