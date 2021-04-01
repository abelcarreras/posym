import pandas as pd
import numpy as np


class CharTable(pd.DataFrame):
    """
    Subclass of DataFrame to add some convenience

    """
    def __init__(self, data, name, rotations, translations, multiplicities):
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
    CharTable({'Ag': pd.Series([+1, +1, +1, +1], index=['E', 'C2', 'i', 'sh']),
               'Bg': pd.Series([+1, -1, +1, -1], index=['E', 'C2', 'i', 'sh']),
               'Au': pd.Series([+1, +1, -1, -1], index=['E', 'C2', 'i', 'sh']),
               'Bu': pd.Series([+1, -1, -1, +1], index=['E', 'C2', 'i', 'sh']),
               },
              name='C2h',
              rotations=['Bg', 'Bg', 'Ag'],  # x, y, z
              translations=['Bu', 'Bu', 'Au'],  # Rx, Ry, Rz
              multiplicities=[1, 1, 1, 1]),

    CharTable({'A1': pd.Series([+1, +1, +1, +1], index=['E', 'C2', 'sv_xz', 'sv_yz']),
               'A2': pd.Series([+1, +1, -1, -1], index=['E', 'C2', 'sv_xz', 'sv_yz']),
               'B1': pd.Series([+1, -1, +1, -1], index=['E', 'C2', 'sv_xz', 'sv_yz']),
               'B2': pd.Series([+1, -1, -1, +1], index=['E', 'C2', 'sv_xz', 'sv_yz']),
               },
              name='C2v',
              rotations=['B2', 'B1', 'A2'],
              translations=['B1', 'B2', 'A1'],
              multiplicities=[1, 1, 1, 1]),

    CharTable({'A1': pd.Series([+1, +1, +1, +1, +1], index=['E', 'C3', 'C2', 'S4', 'sd']),
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

