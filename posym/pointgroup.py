import numpy as np
from posym.ir_tables import ir_table_list


class PointGroup():
    """
    Basic class to basically handle IR-table.

    """
    def __init__(self, group):
        self._group = group

        self._trans_matrix = None
        self._trans_matrix_inv = None

        for table in ir_table_list:
            if group == table.name:
                self._table = table
                return

        raise Exception('{} group not found'.format(group))

    def __str__(self):
        return '{}\n{}'.format(self._table.name, self._table.T)

    @property
    def ir_table(self):
        return self._table

    @property
    def n_sym_elements(self):
        return np.sum(self._table.multiplicities)

    @property
    def n_ir(self):
        return len(self._table)

    @property
    def trans_matrix(self):
        """
        transforms IR to Op

        Op  = Mat * IR

        :return: the transformation matrix
        """
        if self._trans_matrix is None:
            self._trans_matrix = np.array([v for v in self.ir_table.T.values]).T

        return self._trans_matrix

    @property
    def trans_matrix_inv(self):
        """
        transforms Op to IR

        IR = Mat * Op

        :return: the transformation matrix
        """

        if self._trans_matrix_inv is None:
            self._trans_matrix_inv = np.linalg.inv(self.trans_matrix)

        return self._trans_matrix_inv
