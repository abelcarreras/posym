# list all symmetry operations in the original orientation of the molecule
from posym import SymmetryMolecule
from posym.config import Configuration, CustomPerm

# posim configuration
Configuration().scan_steps = 10
Configuration().algorithm = 'exact' # hungarian

# custom permutation
CustomPerm().perm_list = [[2, 0, 3, 1],  # used in calculation
                          [2, 0, 3, 1]]  # used in check

coordinates =[[-0.5, -0.5, 0.0],
              [ 0.5, -0.5, 0.0],
              [-0.5,  0.5, 0.0],
              [ 0.5,  0.5, 0.0]]

symbols = ['H', 'H', 'H', 'H']
group = 'C4'

# print(' structure')
print('\nPoint Group: {}'.format(group))

sm = SymmetryMolecule(group=group, coordinates=coordinates, symbols=symbols, orientation_angles=[-90., 0., 0.])

print(sm.get_point_group(), '\n')
print('Group measure: {:.4f}'.format(sm.measure))
print('Symmetry center: ', sm.center)
print('Group orientation Euler angles zyx (deg): ', sm.orientation_angles)

sm.print_operations_info()
