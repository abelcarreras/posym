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
from posym.ir_tables import ir_table_list


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
        label = 'C{}'.format(down) if up == 1 else 'C^{}_{}'.format(up, down)

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
        rotations = [ 'E', 'A']
        translations = ['E', 'A']
    else:
        rotations = ['E1', 'A']
        translations = ['E1', 'A']

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

    multiplicities = ir_data.multiplicities
    if np.mod(n, 2) == 0:
        multiplicities += [n//2, n//2]
    else:
        multiplicities += [n]

    if n == 2:
        rotations = ['B1', 'B2', 'A2']
        translations = ['B1', 'B2', 'A1']
    elif n == 3 or n == 4:
        rotations = ['E', 'A2']
        translations = ['E', 'A1']
    else:
        rotations = ['E1', 'A2']
        translations = ['E1', 'A1']

    return CharTable('C{}v'.format(n),
                     operations_new,
                     ir_data_new,
                     rotations=rotations,  # x, y, z
                     translations=translations,  # Rx, Ry, Rz
                     multiplicities=multiplicities)


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

        operations_new += [Inversion(label='i'), Reflection(label='sh', axis=[0, 0, 1])]

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


        multiplicities = [1] + [2]*((n-1)//2) + [1, 1] + [2]*((n-1)//2) + [1]

        if n == 2:
            rotations = ['Bu', 'Bu', 'Au']
            translations = ['Bg', 'Bg', 'Ag']
        elif  n == 4:
            rotations = ['Eu', 'Au']
            translations = ['Eg', 'Ag']
        else:
            rotations = ['E1u', 'Au']
            translations = ['E1g', 'Ag']
    else:

        ir_data_new["A'"] = pd.Series(list(ir_data['A']) + [1])
        ir_data_new["A''"] = pd.Series(list(ir_data['A']) + [-1])

        multiplicities = [1] + [2]*((n-1)//2) + [1] + [2]*((n-1)//2)

        operations_new += [Reflection(label='sh', axis=[0, 0, 1])]

        for data in ir_data.keys():
            if data.startswith('E'):
                ir_data_new[data+"'"] = pd.Series(list(ir_data[data]) + [2])
                ir_data_new[data+"''"] = pd.Series(list(ir_data[data]) + [-2])

        for i in range((n-1)//2):
            j = gcd(2*i+1, n)
            up = (2*i+1)//j
            #print('----------------', n, i*2+1, ':', (2*i+1)//j, n//j)
            down = n//j
            if up == 1:
                operations_new += [ImproperRotation(label='S_{}'.format(down), axis=[0, 0, 1], order=down, exp=up)]
            else:
                operations_new += [ImproperRotation(label='S^{}_{}'.format(up, down), axis=[0, 0, 1], order=down, exp=up)]

            for l, data in enumerate(ir_data_new.keys()):
                # real_radical((i + 1) * (k + 1), n)
                sign = np.sign(list(ir_data_new[data])[(n - 1) // 2 + 1])
                element = list(ir_data_new[data])[1+i] * sign
                if data.startswith('E'):
                    try:
                        k = int(data[1])-1
                    except ValueError:
                        k = 0

                    # print([real_radical((k + 1) * (m + 1), n) for m in range(0, n,2)])
                    # element = [real_radical((k + 1) * (m + 1), n) for m in range(0, n, 2)][i] * sign
                    element = real_radical((k + 1) * (2*i + 1), n) * sign

                ir_data_new[data] = pd.Series(list(ir_data_new[data]) + [element])

        if n == 1:
            rotations = ["A'", "A'", "A''"]
            translations = ["A''", "A''", "A'"]
        elif n == 3:
            rotations = ["E'", "A''"]
            translations = ["E''", "A'"]
        else:
            rotations = ["E1'", "A''"]
            translations = ["E1''", "A'"]

    return CharTable('C{}h'.format(n),
                     operations_new,
                     ir_data_new,
                     rotations=rotations,  # x, y, z
                     translations=translations,  # Rx, Ry, Rz
                     multiplicities=multiplicities)

def get_sn(n):

    if np.mod(n, 2) != 0:
        raise Exception('Order of Sn group must have even')

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
                label = 'S{}'.format(down) if up == 1 else 'S^{}_{}'.format(up, down)
                operations.append(ImproperRotation(label=label, axis=[0, 0, 1], order=down, exp=up))
            else:
                label = 'C{}'.format(down) if up == 1 else 'C^{}_{}'.format(up, down)
                operations.append(Rotation(label=label, axis=[0, 0, 1], order=down, exp=up))

        ir_data = {'A': pd.Series([1] * ndim)}
        if r == 1:
            ir_data.update({'B': pd.Series([(-1)**k for k in range(ndim)])})

        for i in range(l):
            label = 'E{}'.format(i+1) if l > 1 else 'E'
            ir_data.update({label: pd.Series([2] + [real_radical((i+1)*(k+1), n) for k in range(ndim-1)])})

        if n == 4:
            rotations = ['E', 'A']
            translations = ['E', 'B']
        else:
            rotations = ['E1', 'A']
            translations = ['E{}'.format(l//2), 'B']

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
                    label = 'S{}'.format(down) if up == 1 else 'S^{}_{}'.format(up, down)
                    operations.append(ImproperRotation(label=label, axis=[0, 0, 1], order=down, exp=up))
            else:
                label = 'C{}'.format(down) if up == 1 else 'C^{}_{}'.format(up, down)
                operations.append(Rotation(label=label, axis=[0, 0, 1], order=down, exp=up))

        ir_data = {'Ag': pd.Series([1] * ndim)}
        traces = [1]
        for i in range(n//2):
            if np.mod(i, 2) == 0:
                traces.append(-1)
            else:
                traces.append(1)

        for i in range(l//2):
            subtrace = np.roll([k+1 for k in range(n//4)], -i)
            label = 'E{}g'.format(i+1) if l > 2 else 'Eg'
            ir_data.update({label: pd.Series([2] + [real_radical(k, n)*(-1)**(k+2*i) for k in subtrace] + [real_radical(k, n)*(-1)**(k+2*i) for k in subtrace[::-1]] + [2])})

        ir_data.update({'Au': pd.Series(traces)})

        for i in range(l//2):
            subtrace = np.roll([k + 1 for k in range(n // 4)], -i)
            label = 'E{}u'.format(i+1) if l > 2 else 'Eu'
            # print(i, '-', [[2] , [(-1)**(m+1) for m, k in enumerate(subtrace)] , [(-1)**(m+l//2+1) for m, k in enumerate(subtrace[::-1])] , [-2]])
            ir_data.update({label: pd.Series([2] +
                                             [real_radical(k, n)*(-1)**(k+2*i) * (-1)**(m+1) for m, k in enumerate(subtrace)] +
                                             [real_radical(k, n)*(-1)**(k+2*i) *(-1)**(m+l//2+1) for m, k in enumerate(subtrace[::-1])] +
                                             [-2])})

        # reorder columns to follow convention
        new_indexing = [0] + list(range(2, l+1, 2)) + [ndim-1] + list(range(1, l+1, 2))
        operations = np.array(operations)[new_indexing].tolist()
        multiplicity = np.array(multiplicity)[new_indexing].tolist()
        for key in ir_data.keys():
            ir_data[key] = ir_data[key][new_indexing]

        if n == 2:
            rotations = ['Ag', 'Ag', 'Ag']
            translations = ['Au', 'Au', 'Au']
        elif n == 6:
            rotations = ['Eg', 'Ag']
            translations = ['Eu', 'Au']
        else:
            rotations = ['E1g', 'Ag']
            translations = ['E1u', 'Au']

    return CharTable('S{}'.format(n),
                     operations,
                     ir_data,
                     rotations=rotations,  # x, y, z
                     translations=translations,  # Rx, Ry, Rz
                     multiplicities=multiplicity)


def get_dn(n):

    if n<2:
        raise Exception('Group not valid')

    if n == 2:

        return CharTable('D2',
                         [Identity(label='E'), Rotation(label='C2', axis=[0, 0, 1], order=2),
                          Rotation(label="C2'", axis=[1, 0, 0], order=2),
                          Rotation(label="C2''", axis=[0, 1, 0], order=2)],
                         {'A1': pd.Series([+1, +1, +1, +1]),
                          'B1': pd.Series([+1, +1, -1, -1]),
                          'B2': pd.Series([+1, -1, -1, +1]),
                          'B3': pd.Series([+1, -1, +1, -1])
                         },
                         rotations=['B3', 'B2', 'B1'],
                         translations=['B3', 'B2', 'B1'],
                         multiplicities=[1, 1, 1, 1])

    ir_data = get_cn(n)
    operations_new = ir_data.operations

    ir_data_new = {}
    ir_data_new['A1'] = pd.Series(list(ir_data['A']) + [ 1])
    ir_data_new['A2'] = pd.Series(list(ir_data['A']) + [-1])
    operations_new += [Rotation(label="C2'", axis=[0, 1, 0], order=2)]

    if 'B' in ir_data:
        ir_data_new['B1'] = pd.Series(list(ir_data['B']) + [ 1, -1])
        ir_data_new['B2'] = pd.Series(list(ir_data['B']) + [-1,  1])

        ir_data_new['A1'] = pd.Series(list(ir_data_new['A1']) + [ 1])
        ir_data_new['A2'] = pd.Series(list(ir_data_new['A2']) + [-1])

        operations_new += [Rotation(label="C2''", axis=[np.sin(np.pi/n), np.cos(np.pi/n), 0], order=2)]

    for data in ir_data.keys():
        if data.startswith('E'):
            ir_data_new[data] = pd.Series(list(ir_data[data]) + [0])

    if 'B' in ir_data:
        for data in ir_data.keys():
            if data.startswith('E'):
                ir_data_new[data] = pd.Series(list(ir_data_new[data]) + [0])

    multiplicities = ir_data.multiplicities
    if np.mod(n, 2) == 0:
        multiplicities += [n//2, n//2]
    else:
        multiplicities += [n]

    if n == 2:
        rotations = ['B2', 'B2', 'A1']
        translations = ['B1', 'B2', 'A1']
    elif n == 3 or n == 4:
        rotations = ['E', 'A2']
        translations = ['E', 'A2']
    else:
        rotations = ['E1', 'A2']
        translations = ['E1', 'A2']

    return CharTable('D{}'.format(n),
                     operations_new,
                     ir_data_new,
                     rotations=rotations,  # x, y, z
                     translations=translations,  # Rx, Ry, Rz
                     multiplicities=multiplicities)


def get_dnh(n):

    ir_data = get_dn(n)
    operations_new = ir_data.operations
    ir_data_new = {}
    ir_data_new_u = {}

    if n == 2:
        return CharTable('D2h',
                         [Identity(label='E'), Rotation(label='C2', axis=[0, 0, 1], order=2),
                          Rotation(label="C2'", axis=[0, 1, 0], order=2), Rotation(label="C2''", axis=[1, 0, 0], order=2),
                          Inversion(label='i'), Reflection(label='sh', axis=[0, 0, 1]),
                          Reflection(label='sv', axis=[1, 0, 0]), Reflection(label='sd', axis=[0, 1, 0])],
                         {'Ag': pd.Series( [+1, +1, +1, +1, +1, +1, +1, +1]),
                          'B1g': pd.Series([+1, +1, -1, -1, +1, +1, -1, -1]),
                          'B2g': pd.Series([+1, -1, -1, +1, +1, -1, +1, -1]),
                          'B3g': pd.Series([+1, -1, +1, -1, +1, -1, -1, +1]),
                          'Au': pd.Series( [+1, +1, +1, +1, -1, -1, -1, -1]),
                          'B1u': pd.Series([+1, +1, -1, -1, -1, -1, +1, +1]),
                          'B2u': pd.Series([+1, -1, -1, +1, -1, +1, -1, +1]),
                          'B3u': pd.Series([+1, -1, +1, -1, -1, +1, +1, -1]),
                          },
                         rotations=['B3g', 'B2g', 'B1g'],
                         translations=['B3u', 'B2u', 'B1u'],
                         multiplicities=[1, 1, 1, 1, 1, 1, 1, 1])

    if np.mod(n, 2) == 0:
        # for even n only

        ir_data_new['A1g'] =   pd.Series(list(ir_data['A1']) + [ 1,  1,  1,  1])
        ir_data_new_u['A1u'] = pd.Series(list(ir_data['A1']) + [-1, -1, -1, -1])

        ir_data_new['A2g'] =   pd.Series(list(ir_data['A2']) + [ 1,  1, -1, -1])
        ir_data_new_u['A2u'] = pd.Series(list(ir_data['A2']) + [-1, -1,  1,  1])

        l = (-1)**(n//2)
        ir_data_new['B1g'] =   pd.Series(list(ir_data['B1']) + [ 1,  1*l,  1*l, -1*l])
        ir_data_new_u['B1u'] = pd.Series(list(ir_data['B1']) + [-1, -1*l, -1*l,  1*l])

        ir_data_new['B2g'] =   pd.Series(list(ir_data['B2']) + [ 1,  1*l,  -1*l,  1*l])
        ir_data_new_u['B2u'] = pd.Series(list(ir_data['B2']) + [-1, -1*l,   1*l, -1*l])

        operations_new += [Inversion(label='i'), Reflection(label='sh', axis=[0, 0, 1]),
                           Reflection(label='s_v', axis=[np.sin(np.pi/n), np.cos(np.pi/n), 0]),
                           Reflection(label='sd', axis=[0, 1, 0])]

        for data in ir_data.keys():
            if data.startswith('E'):
                if len(data[1:]) == 0:
                    k = int((np.mod(1, 2) -0.5)*4)
                else:
                    k = int((np.mod(int(data[1:]), 2) -0.5)*4)
                # print(data+'u', list(ir_data[data]))
                ir_data_new[data+'g'] = pd.Series(list(ir_data[data]) + [2, -k, 0, 0])
                ir_data_new_u[data+'u'] = pd.Series(list(ir_data[data]) + [-2, k, 0, 0])

        ir_data_new.update(ir_data_new_u)

        for i in range((n-1)//2):
            j = gcd(i+1, n)
            up = (i+1)//j
            down = n//j
            if up == 1:
                operations_new += [ImproperRotation(label='S{}'.format(down), axis=[0, 0, 1], order=down, exp=up)]
            else:
                operations_new += [ImproperRotation(label='S^{}_{}'.format(up, down), axis=[0, 0, 1], order=down, exp=up)]

            for data in ir_data_new.keys():
                #print(1+i, (n-1)//2+5)
                element = list(ir_data_new[data])[1+i] * np.sign(list(ir_data_new[data])[(n-1)//2+5])
                #print(data, len(ir_data_new[data]), element)
                ir_data_new[data] = pd.Series(list(ir_data_new[data]) + [element])
            #print('----')

        # swap columns for convention
        l = 2 + n // 2+2
        for kk in range(3):
            value = operations_new.pop(l)
            operations_new.append(value)
            for data in ir_data_new.keys():
                row = list(ir_data_new[data])
                value = row.pop(l)
                row.append(value)
                ir_data_new[data] = pd.Series(row)

        #print(((n-1)//2))
        multiplicities = [1] + [2]*((n-1)//2) + [1]  + [n//2, n//2] + [1] + [2]*((n-1)//2) + [1, n//2, n//2]

        # print(multiplicities)
        if n == 2:
            rotations = ['B3g', 'B2g', 'B1g']
            translations = ['B3u', 'B2u', 'B1u']
        elif  n == 4:
            rotations = ['Eg', 'A2g']
            translations = ['Eu', 'A2u']
        else:
            rotations = ['E1g', 'A2g']
            translations = ['E1u', 'A2u']
    else:

        ir_data_new["A1'"] = pd.Series(list(ir_data['A1']) + [1, 1])
        ir_data_new["A1''"] = pd.Series(list(ir_data['A1']) + [-1, -1])

        ir_data_new["A2'"] = pd.Series(list(ir_data['A2']) + [1, -1])
        ir_data_new["A2''"] = pd.Series(list(ir_data['A2']) + [-1, 1])

        multiplicities = [1] + [2]*((n-1)//2) + [n, 1] + [2]*((n-1)//2) + [n]

        operations_new += [Reflection(label='sh', axis=[0, 0, 1]),
                           Reflection(label='sv', axis=[1, 0, 0])]

        for data in ir_data.keys():
            if data.startswith('E'):
                ir_data_new[data+"'"] = pd.Series(list(ir_data[data]) + [2, 0])
                ir_data_new[data+"''"] = pd.Series(list(ir_data[data]) + [-2, 0])

        for i in range((n-1)//2):
            j = gcd(2*i+1, n)
            up = (2*i+1)//j
            down = n//j
            if up == 1:
                operations_new += [ImproperRotation(label='S_{}'.format(down), axis=[0, 0, 1], order=down, exp=up)]
            else:
                operations_new += [ImproperRotation(label='S^{}_{}'.format(up, down), axis=[0, 0, 1], order=down, exp=up)]

            for data in ir_data_new.keys():
                sign = np.sign(list(ir_data_new[data])[(n - 1) // 2 + 2])
                element = list(ir_data_new[data])[1+i] * sign
                if data.startswith('E'):
                    try:
                        k = int(data[1])-1
                    except ValueError:
                        k = 0

                    # print('E{}'.format(k+1), [real_radical((k + 1) * (m + 1), n) for m in range(0, 7)])
                    # element = [real_radical((k + 1) * (m + 1), n) for m in range(0, n,2)][i] * sign
                    element = real_radical((k + 1) * (2*i + 1), n) * sign

                ir_data_new[data] = pd.Series(list(ir_data_new[data]) + [element])

        # swap columns for convention
        l = 2 + n // 2+1
        for kk in range(1):
            value = operations_new.pop(l)
            operations_new.append(value)
            for data in ir_data_new.keys():
                row = list(ir_data_new[data])
                value = row.pop(l)
                row.append(value)
                ir_data_new[data] = pd.Series(row)

        if n == 1:
            rotations = ["A1''", "A2''", "A2'"]
            translations = ["A1'", "A2'", "A2''"]
        elif n == 3:
            rotations = ["E''", "A2'"]
            translations = ["E'", "A2''"]
        else:
            rotations = ["E1''", "A2'"]
            translations = ["E1'", "A2''"]

    return CharTable('D{}h'.format(n),
                     operations_new,
                     ir_data_new,
                     rotations=rotations,  # x, y, z
                     translations=translations,  # Rx, Ry, Rz
                     multiplicities=multiplicities)

def get_dnd(n):

    ir_data_new = {}

    if n < 2:
        raise Exception('Group not valid')

    if np.mod(n, 2) == 0:
        # for even n only
        ir_data = get_sn(2*n)
        operations_new = ir_data.operations

        ir_data_new['A1'] = pd.Series(list(ir_data['A']) + [ 1,  1])
        ir_data_new['A2'] = pd.Series(list(ir_data['A']) + [-1, -1])

        ir_data_new['B1'] = pd.Series(list(ir_data['B']) + [ 1, -1])
        ir_data_new['B2'] = pd.Series(list(ir_data['B']) + [-1,  1])

        operations_new += [Rotation(label="C2'", axis=[np.sin(np.pi/(2*n)), np.cos(np.pi/(2*n)), 0], order=2),
                           Reflection(label='sd', axis=[0, 1, 0])]

        for data in ir_data.keys():
            if data.startswith('E'):
                ir_data_new[data] = pd.Series(list(ir_data[data]) + [0, 0])

        multiplicities = ir_data.multiplicities + [n, n]

        if n == 2:
            rotations = ['E', 'A2']
            translations = ['E', 'B2']
        else:
            rotations = ['E{}'.format(n-1), 'A2']
            translations = ['E1', 'B2']
    else:
        ir_data = get_dn(n)
        operations_new = ir_data.operations
        ir_data_new_u = {}

        ir_data_new["A1g"] = pd.Series(list(ir_data['A1']) + [1, 1])
        ir_data_new["A2g"] = pd.Series(list(ir_data['A2']) + [1, -1])

        ir_data_new_u["A1u"] = pd.Series(list(ir_data['A1']) + [-1, -1])
        ir_data_new_u["A2u"] = pd.Series(list(ir_data['A2']) + [-1, 1])

        multiplicities = [1] + [2]*((n-1)//2) + [n, 1] + [2]*((n-1)//2) + [n]

        operations_new += [Inversion(label='i'), Reflection(label='sd', axis=[np.cos(np.pi/(2*n)), np.sin(np.pi/(2*n)), 0])]

        for data in ir_data.keys():
            if data.startswith('E'):
                ir_data_new[data+"g"] = pd.Series(list(ir_data[data]) + [2, 0])
                ir_data_new_u[data+"u"] = pd.Series(list(ir_data[data]) + [-2, 0])

        ir_data_new.update(ir_data_new_u)

        for i in range((n-1)//2):
            j = gcd(2*i+1, 2*n)
            up = (2*i+1)//j
            down = 2*n//j
            if up == 1:
                operations_new += [ImproperRotation(label='S{}'.format(down), axis=[0, 0, 1], order=down, exp=up)]
            else:
                operations_new += [ImproperRotation(label='S^{}_{}'.format(up, down), axis=[0, 0, 1], order=down, exp=up)]

            for data in ir_data_new.keys():
                sign = np.sign(list(ir_data_new[data])[(n - 1) // 2 + 2])
                element = list(ir_data_new[data])[1+i] * sign
                if data.startswith('E'):
                    try:
                        k = int(data[1])-1
                    except ValueError:
                        k = 0

                    #element = [real_radical((k + 1) * (m + 1), n) for m in range(0, n//2)][-i-1] * sign
                    element = real_radical((k + 1) * (n // 2 + i + 1), n) * sign

                ir_data_new[data] = pd.Series(list(ir_data_new[data]) + [element])

        # swap columns for convention
        l = 2 + n // 2+1
        for kk in range(1):
            value = operations_new.pop(l)
            operations_new.append(value)
            for data in ir_data_new.keys():
                row = list(ir_data_new[data])
                value = row.pop(l)
                row.append(value)
                ir_data_new[data] = pd.Series(row)

        if n == 1:
            rotations = ["A1g", "A2g", "A2g"]
            translations = ["A1'", "A2'", "A2''"]
        elif n == 3:
            rotations = ["Eg", "A2g"]
            translations = ["Eu", "A2u"]
        else:
            rotations = ["E1g", "A2g"]
            translations = ["E1u", "A2u"]

    return CharTable('D{}d'.format(n),
                     operations_new,
                     ir_data_new,
                     rotations=rotations,  # x, y, z
                     translations=translations,  # Rx, Ry, Rz
                     multiplicities=multiplicities)


def get_table_from_label(label):

    label = label.lower().strip()

    # Check into explicitly defined tables
    for table in ir_table_list:
        if label == table.name.lower():
            return table

    # Generate table
    type = label[0]
    if type in ['c', 'd', 's']:
        code = type[0]
        subtype = label[-1]
        try:
            if subtype in ['d', 'h', 'v']:
                order = int(label[1:-1])
                code += subtype
            else:
                order = int(label[1:])
        except ValueError:
            code = order = None

        functions = {'d': get_dn, 'dd': get_dnd, 'dh': get_dnh,
                     'c': get_cn, 'ch': get_cnh, 'cv':get_cnv,
                     's': get_sn}

        try:
            return functions[code](order)
        except KeyError:
            pass

    raise Exception('Point group label {} not recognized'.format(label))


if __name__ == '__main__':

    pg = get_table_from_label('c1')
    print(pg.name)
    print(pg)
    exit()
    #for i in range(1, 9):
    #    print(get_cnh(i))

    print(get_dnd(2))
    print(get_dnd(3))
    print(get_dnd(4))
    print(get_dnd(5))
    print(get_dnd(6))
    print(get_dnd(7))
    print(get_dnd(8))

    exit()

    print(get_dnh(3))
    print(get_dnh(5))
    print(get_dnh(7))
    print(get_dnh(9))

    exit()
    print(get_dn(5))
    print(get_dnh(5))
    print(get_dnh(6))
    print(get_dnh(7))

    exit()
