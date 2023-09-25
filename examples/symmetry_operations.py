# list all symmetry operations in the original orientation of the molecule
from posym import SymmetryMolecule
import numpy as np


sh6_coor = [[ 0.16290727, -0.36340852,  1.00000000],
            [ 1.47290727, -0.36340852,  0.00000000],
            [ 0.16290727,  0.94659148,  0.00000000],
            [ 0.16290727, -0.36340852,  1.31000000],
            [-1.14709273, -0.36340852,  0.00000000],
            [ 0.16290727, -1.67340852,  0.00000000],
            [ 0.16290727, -0.36340852, -1.31000000]]
sh6_sym = ['S', 'H', 'H', 'H', 'H', 'H', 'H']
group = 'Oh'

# print(' structure')
print('\nPoint Group: {}'.format(group))

sm = SymmetryMolecule(group=group, coordinates=sh6_coor, symbols=sh6_sym)

print(sm.get_point_group())
print('Group measure: ', sm.measure_pos)
print('Symmetry center: ', sm.center)
print('Group orientation Euler angles zyx (deg): ', sm.orientation_angles)

print('\nOperations list\n---------------')
for operation in sm.get_oriented_operations():
    print('Operation object: ', operation)
    print('label:', operation.label)
    try:
        print('Order:', operation.order)
        print('Axis:', operation.axis)
    except AttributeError:
        pass

    print('Operation matrices:')
    for matrix in operation.operation_matrix_list:
        print(np.round(matrix, decimals=3), '\n')
