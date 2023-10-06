# Example of the calculation of continuous measures of symmetry (CSM)
# this shows a calculation of CSM for a equilateral triangular molecule plus a central atom for different point groups
# in this example the geometry is distorted by moving the central atom in the perpendicular direction to the plane
# created by the 3 external atoms
from posym import SymmetryMolecule
from posym.config import Configuration
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time


# remove warnings due to indeterminate Euler angles in some extreme geometries
warnings.simplefilter("ignore", UserWarning)

# define a large pre-scan step to increase speed in expense of accuracy
Configuration().scan_steps = 50

t_1 = time.time()

d_range = np.linspace(0, 5.0, 50)

for group in ['Ci', 'Cs', 'C3h', 'Td']:

    measures = []
    for d in d_range:
        molecule_coor = [[ 1.00000000,   0.00000000,   0.00000000],
                         [-0.50000000,   0.86602540,   0.00000000],
                         [-0.50000000,  -0.86602540,   0.00000000],
                         [0, 0, d]]

        molecule_symbols = ['H', 'H', 'H', 'H']
        sm = SymmetryMolecule(group, molecule_coor, molecule_symbols)

        print(sm.get_point_group())

        print('IR rep: ', sm)
        print('CSM: ', sm.measure)
        print('Center', sm.center)
        print('symmetrized coordinates')
        for c in sm.symmetrized_coordinates:
            print('{:12.8f} {:12.8f} {:12.8f}'.format(*c))

        measures.append(sm.measure)

    plt.plot(d_range, measures, label=group)

plt.xlabel('distance')
plt.ylabel('CSM [Asymmetry]')
plt.legend()
#plt.show()

t_2 = time.time()

print('\n\nTotal time: {:.2f}: '.format(t_2 - t_1))
