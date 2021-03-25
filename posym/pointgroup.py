import numpy as np
from posym.ir_tables import ir_table_list


class PointGroup():
    def __init__(self, group):
        self._group = group

        for table in ir_table_list:
            if group == table.name:
                self._table = table
                return

        raise Exception('{} group not defined'.format(group))

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
