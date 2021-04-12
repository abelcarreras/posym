import pandas as pd
import numpy as np
from posym.operations import Operation
from posym.operations.identity import Identity
from posym.operations.rotation import Rotation
from posym.operations.reflection import Reflection
from posym.operations.inversion import Inversion


def real_radical(m, n):
    return 2 * np.cos(2 * m * np.pi / n)


class Op:
    def __init__(self, label, type, axis=None, order=1):

        if not issubclass(type, Operation):
            raise Exception('Not an Operation')

        self._type = type
        self._axis = axis
        self._order = order
        self._label = label

    @property
    def type(self):
        return self._type

    @property
    def axis(self):
        return self._axis

    @property
    def order(self):
        return self._order

    @property
    def label(self):
        return self._label


class CharTable(pd.DataFrame):
    """
    Subclass of DataFrame to add some convenience

    """
    def __init__(self, name, operations, ir, rotations, translations, multiplicities):

        labels = [o.label for o in operations]

        data = {}
        for key, val in ir.items():
            val.index = labels
            data[key] = val

        super().__init__(data)
        self.name = name

        if np.all([r in data for r in rotations]):
            self.attrs['rotations'] = rotations
        else:
            raise Exception('Rotations representations not in table')

        if np.all([r in data for r in translations]):
            self.attrs['translations'] = translations
        else:
            raise Exception('Translations representations not in table')

        if len(multiplicities) == len(self.index):
            self.attrs['multiplicities'] = multiplicities
        else:
            raise Exception('Multiplicities do not match')

        self.attrs['operations'] = operations

    @property
    def rotations(self):
        return [self[ir] for ir in self.attrs['rotations']]

    @property
    def translations(self):
        return [self[ir] for ir in self.attrs['translations']]

    @property
    def multiplicities(self):
        return self.attrs['multiplicities']

    @property
    def operations(self):
        return self.attrs['operations']



class OldCharTable(pd.DataFrame):
    """
    Subclass of DataFrame to add some convenience

    """
    def __init__(self, data, name, rotations, translations, multiplicities, operations=None):
        super().__init__(data)
        self.name = name

        if np.all([r in data for r in rotations]):
            self.attrs['rotations'] = rotations
        else:
            raise Exception('Rotations representations not in table')

        if np.all([r in data for r in translations]):
            self.attrs['translations'] = translations
        else:
            raise Exception('Translations representations not in table')

        if len(multiplicities) == len(self.index):
            self.attrs['multiplicities'] = multiplicities
        else:
            raise Exception('Multiplicities do not match')

    @property
    def rotations(self):
        return [self[ir] for ir in self.attrs['rotations']]

    @property
    def translations(self):
        return [self[ir] for ir in self.attrs['translations']]

    @property
    def multiplicities(self):
        return self.attrs['multiplicities']


ir_table_list = [
    OldCharTable({'A': pd.Series([+1], index=['E']),
                  },
                 name='C1',
                 rotations=['A', 'A', 'A'],  # x, y, z
                 translations=['A', 'A', 'A'],  # Rx, Ry, Rz
                 multiplicities=[1]),

    OldCharTable({"A'": pd.Series([+1, +1], index=['E', 'sh']),
                  "A''": pd.Series([+1, -1], index=['E', 'sh']),
                  },
                 name='Cs',
                 rotations=["A'", "A'", "A''"],  # x, y, z
                 translations=["A''", "A''", "A'"],  # Rx, Ry, Rz
                 multiplicities=[1, 1]),

    OldCharTable({'Ag': pd.Series([+1, +1], index=['E', 'i']),
                  'Au': pd.Series([+1, -1], index=['E', 'i']),
                  },
                 name='Ci',
                 rotations=['Au', 'Au', 'Au'],  # x, y, z
                 translations=['Ag', 'Ag', 'Ag'],  # Rx, Ry, Rz
                 multiplicities=[1, 1]),

    OldCharTable({'A': pd.Series([+1, +1], index=['E', 'C3']),
                  'E': pd.Series([+2, -1], index=['E', 'C3']),
                  },
                 name='C3',
                 rotations=['E', 'E', 'A'],  # x, y, z
                 translations=['E', 'E', 'A'],  # Rx, Ry, Rz
                 multiplicities=[1, 2]),

    OldCharTable({'A' : pd.Series([+1, +1, +1], index=['E', 'C5', 'C5_2']),
                 'E1': pd.Series([+2, real_radical(2, 5), real_radical(4, 5)], index=['E', 'C5', 'C5_2']),
                 'E2': pd.Series([+2, real_radical(4, 5), real_radical(2, 5)], index=['E', 'C5', 'C5_2']),
                  },
                 name='C5',
                 rotations=['E1', 'E1', 'A'],  # x, y, z
                 translations=['E1', 'E1', 'A'],  # Rx, Ry, Rz
                 multiplicities=[1, 2, 2]),

    CharTable('C2h',
              [Identity(label='E'), Rotation(label='C2', axis=[0, 0, 1], order=2),
               Inversion(label='i'), Reflection(label='sh', axis=[0, 0, 1])],
              {'Ag': pd.Series([+1, +1, +1, +1]),
               'Bg': pd.Series([+1, -1, +1, -1]),
               'Au': pd.Series([+1, +1, -1, -1]),
               'Bu': pd.Series([+1, -1, -1, +1]),
                },
              rotations=['Bg', 'Bg', 'Ag'],  # x, y, z
              translations=['Bu', 'Bu', 'Au'],  # Rx, Ry, Rz
              multiplicities=[1, 1, 1, 1]),
    CharTable('C2v',
              [Identity(label='E'), Rotation(label='C2', axis=[0, 0, 1], order=2),
               Reflection(label='sv_xz', axis=[1, 0, 0]), Reflection(label='sv_yz', axis=[0, 1, 0])],
              {'A1': pd.Series([+1, +1, +1, +1]),
               'A2': pd.Series([+1, +1, -1, -1]),
               'B1': pd.Series([+1, -1, +1, -1]),
               'B2': pd.Series([+1, -1, -1, +1])
               },
              rotations=['B2', 'B1', 'A2'],
              translations=['B1', 'B2', 'A1'],
              multiplicities=[1, 1, 1, 1]),
    OldCharTable({'A1': pd.Series([+1, +1, +1], index=['E', 'C3', 'sv']),
                  'A2': pd.Series([+1, +1, -1], index=['E', 'C3', 'sv']),
                  'E' : pd.Series([+2, -1,  0], index=['E', 'C3', 'sv']),
                  },
                 name='C3v',
                 rotations=['E', 'E', 'A2'],  # x, y, z
                 translations=['E', 'E', 'A1'],  # Rx, Ry, Rz
                 multiplicities=[1, 2, 3]),

    OldCharTable({'A1': pd.Series([+1, +1, +1, +1, +1], index=['E', 'C3', 'C2', 'S4', 'sd']),
                  'A2': pd.Series([+1, +1, +1, -1, -1], index=['E', 'C3', 'C2', 'S4', 'sd']),
                  'E' : pd.Series([+2, -1, +2,  0,  0], index=['E', 'C3', 'C2', 'S4', 'sd']),
                  'T1': pd.Series([+3,  0, -1, +1, -1], index=['E', 'C3', 'C2', 'S4', 'sd']),
                  'T2': pd.Series([+3,  0, -1, -1, +1], index=['E', 'C3', 'C2', 'S4', 'sd']),
                  },
                 name='Td',
                 rotations=['T1', 'T1', 'T1'],
                 translations=['T2', 'T2', 'T2'],
                 multiplicities=[1, 8, 3, 6, 6]),
]

if __name__ == '__main__':

    print(ir_table_list[-1].sort_index().index)
    exit()

    a = np.array([ v for v in ir_table_list[-1].T.values]).T
    print('trans matrix')
    print(a)

    print('t1')
    t1 = np.dot(a, [0, 0, 0, 1, 0])
    print(t1)

    a_i = np.linalg.inv(a)
    print('inverse')
    print(a_i)

    print('t1(IR)')
    print(np.dot(a_i, t1))

    t1t1 = np.multiply(t1, t1)
    print('t1*t1', t1t1)

    e = np.dot(a, [0, 0, 1, 0, 0])
    et1 = np.multiply(e, t1)

    print('t1t1(IR)')
    print(np.dot(a_i, t1t1))
    print('et1')
    print(np.dot(a_i, et1))

