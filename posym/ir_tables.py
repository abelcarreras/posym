import pandas as pd
import numpy as np
from posym.operations import Operation
from posym.operations.identity import Identity
from posym.operations.rotation import Rotation
from posym.operations.reflection import Reflection
from posym.operations.inversion import Inversion
from posym.operations.irotation import ImproperRotation


def real_radical(m, n):
    return 2 * np.cos(2 * m * np.pi / n)


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

    def get_all_operations(self):

        if 'all_operations' in self.attrs:
            return self.attrs['all_operations']

        total_op = list(self.operations)
        for op1 in self.operations:
            for op2 in self.operations:
                opt_list = op1 * op2
                for op in opt_list:
                    if op not in total_op:
                        total_op.append(op)

        repeat = True
        while repeat is True:
            repeat = False
            for op1 in tuple(total_op):
                for op2 in tuple(total_op):
                    opt_list = op1 * op2
                    for op in opt_list:
                        if op not in total_op:
                            total_op.append(op)
                            repeat = True

        operation_dict = {}
        for op in total_op:
            if not op.label in operation_dict:
                operation_dict[op.label] = [op]
            else:
                operation_dict[op.label].append(op)

        self.attrs['all_operations'] = operation_dict

        return self.attrs['all_operations']

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

    @property
    def ir_degeneracies(self):
        return self.T['E'].values


ir_table_list = [

    CharTable('C1',
              [Identity(label='E')],
              {'A': pd.Series([+1]),
               },
              rotations=['A', 'A', 'A'],  # x, y, z
              translations=['A', 'A', 'A'],  # Rx, Ry, Rz
              multiplicities=[1]),

    CharTable('Cs',
              [Identity(label='E'), Reflection(label='sh', axis=[0, 0, 1])],
              {"A'": pd.Series([+1, +1]),
               "A''": pd.Series([+1, -1])
               },
              rotations=["A'", "A'", "A''"],  # x, y, z
              translations=["A''", "A''", "A'"],  # Rx, Ry, Rz
              multiplicities=[1, 1]),

    CharTable('Ci',
              [Identity(label='E'), Inversion(label='i')],
              {'Ag': pd.Series([+1, +1]),
               'Au': pd.Series([+1, -1])
               },
              rotations=['Au', 'Au', 'Au'],  # x, y, z
              translations=['Ag', 'Ag', 'Ag'],  # Rx, Ry, Rz
              multiplicities=[1, 1]),

    CharTable('C3',
              [Identity(label='E'), Rotation(label='C3', axis=[0, 0, 1], order=3)],
              {'A': pd.Series([+1, +1]),
               'E': pd.Series([+2, -1])
               },
              rotations=['E', 'E', 'A'],  # x, y, z
              translations=['E', 'E', 'A'],  # Rx, Ry, Rz
              multiplicities=[1, 2]),

    CharTable('C5',
              [Identity(label='E'), Rotation(label='C5', axis=[0, 0, 1], order=5),
               Rotation(label='C5_2', axis=[0, 0, 1], order=5)],
              {'A' : pd.Series([+1, +1, +1]),
               'E1': pd.Series([+2, real_radical(2, 5), real_radical(4, 5)]),
               'E2': pd.Series([+2, real_radical(4, 5), real_radical(2, 5)])
               },
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

    CharTable('C3v',
              [Identity(label='E'), Rotation(label='C3', axis=[0, 0, 1], order=3),
               Reflection(label='sv', axis=[1, 0, 0])],
              {'A1': pd.Series([+1, +1, +1]),
               'A2': pd.Series([+1, +1, -1]),
               'E':  pd.Series([+2, -1,  0])
               },
              rotations=['E', 'E', 'A2'],
              translations=['E', 'E', 'A1'],
              multiplicities=[1, 2, 3]),

    CharTable('Td',
              [Identity(label='E'), Rotation(label='C3', axis=[0, 0, 1], order=3),
               Rotation(label='C2', axis=[np.sqrt(2/9), 0, 1/3], order=2),
               ImproperRotation(label='S4', axis=[np.sqrt(2/9), 0, 1/3], order=4),
               Reflection(label='sd', axis=[0, 1, 0])],
              {'A1': pd.Series([+1, +1, +1, +1, +1]),
               'A2': pd.Series([+1, +1, +1, -1, -1]),
               'E' : pd.Series([+2, -1, +2,  0,  0]),
               'T1': pd.Series([+3,  0, -1, +1, -1]),
               'T2': pd.Series([+3,  0, -1, -1, +1])
               },
              rotations=['T1', 'T1', 'T1'],
              translations=['T2', 'T2', 'T2'],
              multiplicities=[1, 8, 3, 6, 6]),
]

if __name__ == '__main__':


    ct = ir_table_list[-1]
    print(ct.ir_degeneracy)

    exit()



    print(ir_table_list[-1].sort_index().index)
    # exit()

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

    print('--------')

    td = ir_table_list[-1]

    print(0, 0, 0)
    print(0, 0, 1)
    print(np.sqrt(8 / 9), 0, -1 / 3)
    print(0, 1, 0)
    exit()

    #[Identity(label='E'), Rotation(label='C3', axis=[0, 0, 1], order=3),
    # Rotation(label='C2', axis=[np.sqrt(8 / 9), 0, -1 / 3], order=2),
    # ImproperRotation(label='S4', axis=[np.sqrt(8 / 9), 0, -1 / 3], order=4),
