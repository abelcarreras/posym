import pandas as pd
import numpy as np
from posym.operations.identity import Identity
from posym.operations.rotation import Rotation
from posym.operations.reflection import Reflection
from posym.operations.inversion import Inversion
from posym.operations.irotation import ImproperRotation


def real_radical(m, n):
    return 2 * np.cos(2 * m * np.pi / n)


class CharTable(pd.DataFrame):
    """
    Subclass of DataFrame that contains the symmetry table data

    """
    def __init__(self, name, operations, ir, rotations, translations, multiplicities, generator_indices=()):

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

        # define generators
        generator_list = []
        for i in generator_indices:
            generator = operations[i]
            generator._gen_rep = [generator]  # set itself as generators representation
            generator_list.append(generator)

        self.attrs['generators'] = generator_list

    def __str__(self):

        table = self.copy()
        new_index = []
        for op_label, m in zip(table.index, table.attrs['multiplicities']):
            if m > 1:
                new_index.append('{}{}'.format(m, op_label))
            else:
                new_index.append('{}'.format(op_label))

        table.index = new_index
        formatted_data = {k: "{:,.4f}".format for k in table.T.columns}
        # formatted_data = {k: "{:}".format for k in table.T.columns}  # no format

        return table.T.to_string(formatters=formatted_data)

    def get_all_operations(self):

        if 'all_operations' in self.attrs:
            return self.attrs['all_operations']

        # generate all operations of the group and store its relation in connection
        def add_layer(opt_list, opi):
            for gen_op in self.generators:
                op_p = gen_op * opi
                if not op_p in opt_list:
                    opt_list.append(op_p)

        def generate_operations(opt_list):

            while True:
                list_ini = list(opt_list)
                for op in opt_list:
                    add_layer(opt_list, op)

                if len(list_ini) == len(opt_list):
                    break

        opt_list = [self.operations[0]]
        generate_operations(opt_list)

        """
        print('n_operations:', len(opt_list))

        print('\nOperators from generators')
        for op in opt_list:
            try:
                print('connect: ', op.label, op.axis, op.order, op.exp, op)
            except:
                try:
                    print('connect: ', op.label, op.axis)
                except:
                    print('connect: ', op.label)
        """

        # set label to operations according to classes
        operation_dict = {}
        for op_ref in self.operations:
            # print('ref: ', op_ref.label, '               ', op_ref)
            operation_dict[op_ref.label] = []
            for op in opt_list:
                if op.get_type() == op_ref.get_type():
                    for op_b in opt_list:
                        prod = op_b * op * op_b.inverse()
                        if (op_ref == prod) or (op_ref == prod.inverse()):
                            #print('   found!', op)
                            #print('           check:', op_b * op * op_b.inverse())
                            op._label = op_ref.label
                            operation_dict[op_ref.label].append(op)
                            break

        # add gen_rep to operations
        for op_ref in self.operations:
            for op in opt_list:
                if op_ref == op:
                    op_ref._gen_rep = op._gen_rep

        self.attrs['all_operations'] = operation_dict

        return self.attrs['all_operations']

    @property
    def rotations(self):
        return [ir for ir in self.attrs['rotations']]

    @property
    def translations(self):
        return [ir for ir in self.attrs['translations']]

    @property
    def multiplicities(self):
        return self.attrs['multiplicities']

    @property
    def operations(self):
        return self.attrs['operations']

    @property
    def generators(self):
        return self.attrs['generators']

    @property
    def ir_degeneracies(self):
        return self.T['E'].values


ir_table_list = [

    CharTable('Cs',
              [Identity(label='E'), Reflection(label='sh', axis=[0, 0, 1])],
              {"A'": pd.Series([+1, +1]),
               "A''": pd.Series([+1, -1])
               },
              rotations=["A'", "A'", "A''"],  # x, y, z
              translations=["A''", "A''", "A'"],  # Rx, Ry, Rz
              multiplicities=[1, 1],
              generator_indices=[1]),

    CharTable('Ci',
              [Identity(label='E'), Inversion(label='i')],
              {'Ag': pd.Series([+1, +1]),
               'Au': pd.Series([+1, -1])
               },
              rotations=['Au', 'Au', 'Au'],  # x, y, z
              translations=['Ag', 'Ag', 'Ag'],  # Rx, Ry, Rz
              multiplicities=[1, 1],
              generator_indices=[1]),

    CharTable('T',
              [Identity(label='E'), Rotation(label='C3', axis=[0, 0, 1], order=3),
               Rotation(label='C2', axis=[np.sqrt(2 / 9), 0, 1 / 3], order=2)],
              {'A': pd.Series([+1, +1, +1]),
               'E': pd.Series([+2, -1, +2]),
               'T': pd.Series([+3,  0, -1])
               },
              rotations=['T'],
              translations=['T'],
              multiplicities=[1, 8, 3],
              generator_indices=[1, 2]),

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
              rotations=['T1'],
              translations=['T2'],
              multiplicities=[1, 8, 3, 6, 6],
              generator_indices=[1, 3]),

    CharTable('Th',
              [Identity(label='E'), Rotation(label='C3', axis=[0, 0, 1], order=3),
               Rotation(label='C2', axis=[np.sqrt(2/9), 0, 1/3], order=2),
               Inversion(label='i'),
               ImproperRotation(label='S6', axis=[0, 0, 1], order=6),
               Reflection(label='sh', axis=[np.sqrt(2/9), 0, 1/3])],
              {'Ag': pd.Series([+1, +1, +1, +1, +1, +1]),
               'Eg': pd.Series([+2, -1, +2, +2, -1, +2]),
               'Tg': pd.Series([+3,  0, -1, +3,  0, -1]),
               'Au': pd.Series([+1, +1, +1, -1, -1, -1]),
               'Eu': pd.Series([+2, -1, +2, -2, +1, -2]),
               'Tu': pd.Series([+3,  0, -1, -3,  0, +1])
               },
              rotations=['Tg'],
              translations=['Tu'],
              multiplicities=[1, 8, 3, 1, 8, 3],
              generator_indices=[2, 4]),

    CharTable('O',
              [Identity(label='E'),
               Rotation(label='C3', axis=[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], order=3),
               Rotation(label='C2', axis=[0, 0, 1], order=2, exp=1),
               Rotation(label='C4', axis=[0, 0, 1], order=4),
               Rotation(label="C2'", axis=[1/np.sqrt(2), 1/np.sqrt(2), 0], order=2)],
              {'A1': pd.Series([+1, +1, +1, +1, +1]),
               'A2': pd.Series([+1, +1, +1, -1, -1]),
               'E' : pd.Series([+2, -1, +2,  0,  0]),
               'T1': pd.Series([+3,  0, -1, +1, -1]),
               'T2': pd.Series([+3,  0, -1, -1, +1])
               },
              rotations=['T1'],
              translations=['T1'],
              multiplicities=[1, 8, 3, 6, 6],
              generator_indices=[1, 3]),

    CharTable('Oh',
              [Identity(label='E'),
               Rotation(label='C3', axis=[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], order=3),
               Rotation(label='C2', axis=[0, 0, 1], order=2),
               Rotation(label='C4', axis=[0, 0, 1], order=4),
               Rotation(label="C2'", axis=[1/np.sqrt(2), 1/np.sqrt(2), 0], order=2),
               Inversion(label='i'),
               ImproperRotation(label='S6', axis=[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], order=6),
               Reflection(label='sh', axis=[0, 0, 1]),
               ImproperRotation(label='S4', axis=[0, 0, 1], order=4),
               Reflection(label="sd", axis=[1/np.sqrt(2), 1/np.sqrt(2), 0]),
               ],
              {'A1g': pd.Series([+1, +1, +1, +1, +1, +1, +1, +1, +1, +1]),
               'A2g': pd.Series([+1, +1, +1, -1, -1, +1, +1, +1, -1, -1]),
               'Eg' : pd.Series([+2, -1, +2,  0,  0, +2, -1, +2,  0,  0]),
               'T1g': pd.Series([+3,  0, -1, +1, -1, +3,  0, -1, +1, -1]),
               'T2g': pd.Series([+3,  0, -1, -1, +1, +3,  0, -1, -1, +1]),
               'A1u': pd.Series([+1, +1, +1, +1, +1, -1, -1, -1, -1, -1]),
               'A2u': pd.Series([+1, +1, +1, -1, -1, -1, -1, -1, +1, +1]),
               'Eu' : pd.Series([+2, -1, +2,  0,  0, -2, +1, -2,  0,  0]),
               'T1u': pd.Series([+3,  0, -1, +1, -1, -3,  0, +1, -1, +1]),
               'T2u': pd.Series([+3,  0, -1, -1, +1, -3,  0, +1, +1, -1]),
               },
              rotations=['T1g'],
              translations=['T1u'],
              multiplicities=[1, 8, 3, 6, 6, 1, 8, 3, 6, 6],
              generator_indices=[3, 6]),

    CharTable('I',
              [Identity(label='E'),
               Rotation(label='C5', axis=[0, -1, (np.sqrt(5) + 1)/2], order=5),
               Rotation(label='C^2_5', axis=[0, -1, (np.sqrt(5) + 1)/2], order=5, exp=2),
               Rotation(label='C3', axis=[1-(np.sqrt(5) + 1)/2, 0, (np.sqrt(5) + 1)/2], order=3),
               Rotation(label="C2'", axis=[0, 0, 1], order=2)],
              {'A' : pd.Series([+1, +1, +1, +1, +1]),
               'T1': pd.Series([+3,  real_radical(1, 10),  real_radical(3, 10),  0, -1]),
               'T2': pd.Series([+3,  real_radical(3, 10),  real_radical(1, 10),  0, -1]),
               'G' : pd.Series([+4, -1, -1, +1,  0]),
               'H' : pd.Series([+5,  0,  0, -1, +1])
               },
              rotations=['T1'],
              translations=['T1'],
              multiplicities=[1, 12, 12, 20, 15],
              generator_indices=[1, 3]),

    CharTable('Ih',
              [Identity(label='E'),
               Rotation(label='C5', axis=[0, -1, (np.sqrt(5) + 1) / 2], order=5),
               Rotation(label='C^2_5', axis=[0, -1, (np.sqrt(5) + 1) / 2], order=5, exp=2),
               Rotation(label='C3', axis=[1 - (np.sqrt(5) + 1) / 2, 0, (np.sqrt(5) + 1) / 2], order=3),
               Rotation(label="C2'", axis=[0, 0, 1], order=2),
               Inversion(label='i'),
               ImproperRotation(label='S10', axis=[0, -1, (np.sqrt(5) + 1) / 2], order=10),
               ImproperRotation(label='S^3_10', axis=[0, -1, (np.sqrt(5) + 1) / 2], order=10, exp=3),
               ImproperRotation(label='S6', axis=[1 - (np.sqrt(5) + 1) / 2, 0, (np.sqrt(5) + 1) / 2], order=6),
               Reflection(label='sh', axis=[1, 0, 0])],
              {'Ag' : pd.Series([+1, +1, +1, +1, +1, +1, +1, +1, +1, +1]),
               'T1g': pd.Series([+3, real_radical(1, 10), real_radical(3, 10), 0, -1, +3, real_radical(3, 10), real_radical(1, 10), 0, -1,]),
               'T2g': pd.Series([+3, real_radical(3, 10), real_radical(1, 10), 0, -1, +3, real_radical(1, 10), real_radical(3, 10), 0, -1]),
               'Gg' : pd.Series([+4, -1, -1, +1, 0, +4, -1, -1, +1, 0]),
               'Hg' : pd.Series([+5, 0, 0, -1, +1, +5, 0, 0, -1, +1]),
               'Au' : pd.Series([+1, +1, +1, +1, +1, -1, -1, -1, -1, -1]),
               'T1u': pd.Series([+3, real_radical(1, 10), real_radical(3, 10), 0, -1, -3, -real_radical(3, 10), -real_radical(1, 10), 0, +1]),
               'T2u': pd.Series([+3, real_radical(3, 10), real_radical(1, 10), 0, -1, -3, -real_radical(1, 10), -real_radical(3, 10), 0, +1]),
               'Gu' : pd.Series([+4, -1, -1, +1,  0, -4, +1, +1, -1,  0]),
               'Hu' : pd.Series([+5,  0,  0, -1, +1, -5,  0,  0, +1, -1]),
               },
              rotations=['T1g'],
              translations=['T1u'],
              multiplicities=[1, 12, 12, 20, 15, 1, 12, 12, 20, 15],
              generator_indices=[1, 8]),
]

if __name__ == '__main__':


    for pg in ir_table_list[4:5]:
        print(pg.name)
        print(pg)
        print(len(pg.get_all_operations()))
        for op in pg.get_all_operations():
            print(len(pg.get_all_operations()[op]), op)

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

    # [Identity(label='E'), Rotation(label='C3', axis=[0, 0, 1], order=3),
    # Rotation(label='C2', axis=[np.sqrt(8 / 9), 0, -1 / 3], order=2),
    # ImproperRotation(label='S4', axis=[np.sqrt(8 / 9), 0, -1 / 3], order=4),
