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
            if group.upper() == table.name.upper():
                self._table = table
                return

        raise Exception('{} group not found'.format(group))

    def __hash__(self):
        return hash(self._group)

    def __eq__(self, other):
        return self.label == other.label

    def __str__(self):
        return '{}\n{}'.format(self._table.name, self._table.T.to_string())

    def get_all_operations(self):
        return self._table.get_all_operations()

    @property
    def group(self):
        return self._group

    @property
    def operations(self):
        return self._table.operations

    @property
    def ir_table(self):
        return self._table

    @property
    def op_labels(self):
        return self._table.index

    @property
    def ir_labels(self):
        return self._table.T.index

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
            self._trans_matrix = self._trans_matrix/self._trans_matrix[0, :]  # normalization

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

    @property
    def order(self):
        return np.sum(self._table.multiplicities)

    @property
    def ir_degeneracies(self):
        return self._table.ir_degeneracies

    @property
    def label(self):
        return self._group
