# This is under contruction and not yet in use

import numpy as np
from math import gcd
import pandas as pd
from posym.operations.identity import Identity
from posym.operations.rotation import Rotation
from posym.operations.reflection import Reflection
from posym.operations.inversion import Inversion
from posym.operations.irotation import ImproperRotation
from posym.ir_tables import CharTable
from posym.ir_tables import real_radical


def get_cn(n):

    r = np.clip(np.mod(n-1, 2), 0, 1)
    l = (n-r-1)//2
    ndim = 1 + l + r

    multiplicity = [1] + [2] * l + [1] * r

    operations = [Identity(label='E')]
    for i in range(n//2):
        j = gcd(i+1, n)
        up = (i+1)//j
        down = n//j
        label = 'C_{}'.format(down) if up == 1 else 'C^{}_{}'.format(up, down)

        operations.append(Rotation(label=label, axis=[0, 0, 1], order=down, exp=up))

    ir_data = {'A': pd.Series([1] * ndim)}
    if r == 1:
        ir_data.update({'B': pd.Series([(-1)**k for k in range(ndim)])})

    for i in range(l):
        label = 'E{}'.format(i+1) if l > 1 else 'E'
        ir_data.update({label: pd.Series([2] + [real_radical((i+1)*(k+1), n) for k in range(ndim-1)])})

    if n == 1:
        rotations = ['A', 'A', 'A']
        translations = ['A', 'A', 'A']
    elif n == 2:
        rotations = ['B', 'B', 'A']
        translations = ['B', 'B', 'A']
    elif n == 3 or n == 4:
        rotations = ['E', 'E', 'A']
        translations = ['E', 'E', 'A']
    else:
        rotations = ['E1', 'E1', 'A']
        translations = ['E1', 'E1', 'A']

    name = 'C{}'.format(n)
    #return name, operations, ir_data, rotations, translations, multiplicity
    return CharTable('C{}'.format(n),
                     operations,
                     ir_data,
                     rotations=rotations,  # x, y, z
                     translations=translations,  # Rx, Ry, Rz
                     multiplicities=multiplicity)


def get_cnv(n):

    if n<2:
        raise Exception('Group not valid')

    ir_data = get_cn(n)
    operations_new = ir_data.operations

    ir_data_new = {}
    ir_data_new['A1'] = pd.Series(list(ir_data['A']) + [ 1])
    ir_data_new['A2'] = pd.Series(list(ir_data['A']) + [-1])
    operations_new += [Reflection(label='sv_xz', axis=[0, 1, 0])]

    if 'B' in ir_data:
        ir_data_new['B1'] = pd.Series(list(ir_data['B']) + [ 1, -1])
        ir_data_new['B2'] = pd.Series(list(ir_data['B']) + [-1,  1])

        ir_data_new['A1'] = pd.Series(list(ir_data_new['A1']) + [ 1])
        ir_data_new['A2'] = pd.Series(list(ir_data_new['A2']) + [-1])

        operations_new += [Reflection(label='sd_yz', axis=[1, 0, 0])]

    for data in ir_data.keys():
        if data.startswith('E'):
            ir_data_new[data] = pd.Series(list(ir_data[data]) + [0])

    if 'B' in ir_data:
        for data in ir_data.keys():
            if data.startswith('E'):
                ir_data_new[data] = pd.Series(list(ir_data_new[data]) + [0])

    multiplicites = ir_data.multiplicities
    if np.mod(n, 2) == 0:
        multiplicites += [n//2, n//2]
    else:
        multiplicites += [n]

    if n == 2:
        rotations = ['B2', 'B2', 'A1']
        translations = ['B1', 'B2', 'A1']
    elif n == 3 or n == 4:
        rotations = ['E', 'E', 'A2']
        translations = ['E', 'E', 'A1']
    else:
        rotations = ['E1', 'E1', 'A2']
        translations = ['E1', 'E1', 'A1']

    return CharTable('C{}v'.format(n),
                     operations_new,
                     ir_data_new,
                     rotations=rotations,  # x, y, z
                     translations=translations,  # Rx, Ry, Rz
                     multiplicities=multiplicites)


def get_cnh(n):

    ir_data = get_cn(n)
    operations_new = ir_data.operations
    ir_data_new = {}
    ir_data_new_u = {}

    if np.mod(n, 2) == 0:
        # for even n only

        ir_data_new['Ag'] = pd.Series(list(ir_data['A']) + [ 1, 1])
        ir_data_new_u['Au'] = pd.Series(list(ir_data['A']) + [-1, -1])

        l = (-1)**(n//2)
        ir_data_new['Bg'] = pd.Series(list(ir_data['B']) + [ 1, 1*l])
        ir_data_new_u['Bu'] = pd.Series(list(ir_data['B']) + [-1, -1*l])

        operations_new += [Inversion(label='i'), Reflection(label='s_h', axis=[0, 0, 1])]

        for data in ir_data.keys():
            if data.startswith('E'):
                if len(data[1:]) == 0:
                    k = int((np.mod(1, 2) -0.5)*4)
                else:
                    k = int((np.mod(int(data[1:]), 2) -0.5)*4)
                # print(data+'u', list(ir_data[data]))
                ir_data_new[data+'g'] = pd.Series(list(ir_data[data]) + [2, -k])
                ir_data_new_u[data+'u'] = pd.Series(list(ir_data[data]) + [-2, k])

        ir_data_new.update(ir_data_new_u)

        for i in range((n-1)//2):
            j = gcd(i+1, n)
            up = (i+1)//j
            down = n//j
            if up == 1:
                operations_new += [ImproperRotation(label='S_{}'.format(down), axis=[0, 0, 1], order=down, exp=up)]
            else:
                operations_new += [ImproperRotation(label='S^{}_{}'.format(up, down), axis=[0, 0, 1], order=down, exp=up)]

            for data in ir_data_new.keys():
                element = list(ir_data_new[data])[1+i] * np.sign(list(ir_data_new[data])[(n-1)//2+3])
                ir_data_new[data] = pd.Series(list(ir_data_new[data]) + [element])

        l = 2 + n // 2
        # swap columns for convention
        value = operations_new.pop(l)
        operations_new.append(value)
        for data in ir_data_new.keys():
            row = list(ir_data_new[data])
            value = row.pop(l)
            row.append(value)
            ir_data_new[data] = pd.Series(row)


        multiplicites = [1] + [2]*((n-1)//2) + [1, 1] + [2]*((n-1)//2) + [1]

        if n == 2:
            rotations = ['Bu', 'Bu', 'Au']
            translations = ['Bg', 'Bg', 'Ag']
        elif  n == 4:
            rotations = ['Eu', 'Eu', 'Au']
            translations = ['Eg', 'Eg', 'Ag']
        else:
            rotations = ['E1u', 'E1u', 'Au']
            translations = ['E1g', 'E1g', 'Ag']
    else:

        ir_data_new["A'"] = pd.Series(list(ir_data['A']) + [1])
        ir_data_new["A''"] = pd.Series(list(ir_data['A']) + [-1])

        multiplicites = [1] + [2]*((n-1)//2) + [1] + [2]*((n-1)//2)

        operations_new += [Reflection(label='s_h', axis=[0, 0, 1])]

        for data in ir_data.keys():
            if data.startswith('E'):
                ir_data_new[data+"'"] = pd.Series(list(ir_data[data]) + [2])
                ir_data_new[data+"''"] = pd.Series(list(ir_data[data]) + [-2])

        for i in range((n-1)//2):
            j = gcd(i+1, n)
            up = (i+1)//j
            down = n//j
            if up == 1:
                operations_new += [ImproperRotation(label='S_{}'.format(down), axis=[0, 0, 1], order=down, exp=up)]
            else:
                operations_new += [ImproperRotation(label='S^{}_{}'.format(up, down), axis=[0, 0, 1], order=down, exp=up)]

            for data in ir_data_new.keys():
                element = list(ir_data_new[data])[1+i] * np.sign(list(ir_data_new[data])[(n-1)//2+1])
                ir_data_new[data] = pd.Series(list(ir_data_new[data]) + [element])

        if n == 1:
            rotations = ["A'", "A'", "A''"]
            translations = ["A''", "A''", "A'"]
        elif n == 3:
            rotations = ["E'", "E'", "A''"]
            translations = ["E''", "E''", "A'"]
        else:
            rotations = ["E1'", "E1'", "A''"]
            translations = ["E1''", "E1''", "A'"]

    return CharTable('C{}h'.format(n),
                     operations_new,
                     ir_data_new,
                     rotations=rotations,  # x, y, z
                     translations=translations,  # Rx, Ry, Rz
                     multiplicities=multiplicites)

def get_sn(n):

    if np.mod(n, 2) != 0:
        raise Exception('n must be even')

    if np.mod(n, 4) == 0:

        r = np.clip(np.mod(n-1, 2), 0, 1)
        l = (n-r-1)//2
        ndim = 1 + l + r

        multiplicity = [1] + [2] * l + [1] * r

        operations = [Identity(label='E')]
        for i in range(n//2):
            j = gcd(i+1, n)
            up = (i+1)//j
            down = n//j
            if np.mod(i, 2) == 0:
                label = 'S_{}'.format(down) if up == 1 else 'S^{}_{}'.format(up, down)
                operations.append(ImproperRotation(label=label, axis=[0, 0, 1], order=down, exp=up))
            else:
                label = 'C_{}'.format(down) if up == 1 else 'C^{}_{}'.format(up, down)
                operations.append(Rotation(label=label, axis=[0, 0, 1], order=down, exp=up))

        ir_data = {'A': pd.Series([1] * ndim)}
        if r == 1:
            ir_data.update({'B': pd.Series([(-1)**k for k in range(ndim)])})

        for i in range(l):
            label = 'E{}'.format(i+1) if l > 1 else 'E'
            ir_data.update({label: pd.Series([2] + [real_radical((i+1)*(k+1), n) for k in range(ndim-1)])})

        if n == 4:
            rotations = ['E', 'E', 'A']
            translations = ['E', 'E', 'B']
        else:
            rotations = ['E1', 'E1', 'A']
            translations = ['E{}'.format(l//2), 'E{}'.format(l//2), 'B']

    else:

        r = np.clip(np.mod(n-1, 2), 0, 1)
        l = (n-r-1)//2
        ndim = 1 + l + r

        multiplicity = [1] + [2] * l + [1] * r

        operations = [Identity(label='E')]
        for i in range(n//2):
            j = gcd(i+1, n)
            up = (i+1)//j
            down = n//j
            if np.mod(i, 2) == 0:
                if up == 1 and down == 2:
                    label = 'i'.format(down)
                    operations.append(Inversion(label=label))
                else:
                    label = 'S_{}'.format(down) if up == 1 else 'S^{}_{}'.format(up, down)
                    operations.append(ImproperRotation(label=label, axis=[0, 0, 1], order=down, exp=up))
            else:
                label = 'C_{}'.format(down) if up == 1 else 'C^{}_{}'.format(up, down)
                operations.append(Rotation(label=label, axis=[0, 0, 1], order=down, exp=up))

        ir_data = {'Ag': pd.Series([1] * ndim)}
        traces = [1]
        for i in range(n//2):
            if np.mod(i, 2) == 0:
                traces.append(-1)
            else:
                traces.append(1)

        ir_data.update({'Au': pd.Series(traces)})

        for i in range(0, l//2):
            subtrace = np.roll([k+1 for k in range(n//4)], -i)
            label = 'E{}g'.format(i+1) if l > 2 else 'Eg'
            ir_data.update({label: pd.Series([2] + [real_radical(k, n)*(-1)**(k+2*i) for k in subtrace] + [real_radical(k, n)*(-1)**(k+2*i) for k in subtrace[::-1]] + [2])})
            label = 'E{}u'.format(i+1) if l > 2 else 'Eu'
            ir_data.update({label: pd.Series([2] + [real_radical(k, n)*(-1)**(k+2*i + k//2) for k in subtrace] + [real_radical(k, n)*(-1)**(k+2*i + k//2 + 1) for k in subtrace[::-1]] + [-2])})

        if n == 2:
            rotations = ['Ag', 'Ag', 'Ag']
            translations = ['Au', 'Au', 'Au']
        elif n == 6:
            rotations = ['Eg', 'Eg', 'Ag']
            translations = ['Eu', 'Eu', 'Au']
        else:
            rotations = ['E1g', 'E1g', 'Ag']
            translations = ['E1u', 'E1u', 'Au']

    return CharTable('S{}'.format(n),
                     operations,
                     ir_data,
                     rotations=rotations,  # x, y, z
                     translations=translations,  # Rx, Ry, Rz
                     multiplicities=multiplicity)


if __name__ == '__main__':

    #for i in range(1, 9):
    #    print(get_cnh(i))

    print(get_sn(2))
    print(get_sn(4))
    print(get_sn(6))
    print(get_sn(8))
    print(get_sn(10))
    print(get_sn(12))
    print(get_sn(14))

    exit()
    get_cn(2)
    get_cn(3)
    get_cn(4)
    get_cn(5)
    get_cn(6)
    get_cn(7)
    get_cn(24)
