from pyqchem import get_output_from_qchem, Structure, QchemInput
from pyqchem.parsers.parser_frequencies import basic_frequencies
from posym import PointGroup, SymmetryModes
import matplotlib.pyplot as plt
import numpy as np


m_measure = []
for x_coor in np.arange(-1.0, 1.0, 0.02):
    water = [[x_coor,           0.00000000e+00,  2.40297090e-01],
             [-1.43261539e+00, -1.75444785e-16, -9.61188362e-01],
             [1.43261539e+00,   1.75444785e-16, -9.61188362e-01]]
    water = np.array(water) * 0.5

    molecule_water = Structure(coordinates=water,
                               symbols=['O', 'H', 'H'],
                               charge=0,
                               multiplicity=1)

    dicloro = [[ 2.1437,  0.1015, -0.0002],
               [-2.1439, -0.1011, -0.0002],
               [ 0.5135 + x_coor/10, -0.4232, 0.0002],
               [-0.5132 + x_coor/10,  0.4227, 0.0002],
               [ 0.4242, -1.5014, 0.0001],
               [-0.4237,  1.5009, 0.0001]]
    dcl_symbols = ['Cl', 'Cl', 'C', 'C', 'H', 'H']

    molecule_ch2cl2 = Structure(coordinates=dicloro,
                                symbols=dcl_symbols,
                                charge=0,
                                multiplicity=1)

    qc_input = QchemInput(molecule_water,
                          jobtype='freq',
                          exchange='hf',
                          basis='6-31G',
                          # sym_ignore=True,
                          )

    parsed_data, ee = get_output_from_qchem(qc_input, parser=basic_frequencies, read_fchk=True)

    molecule_coor = np.array(ee['structure'].get_coordinates())
    molecule_symbols = np.array(ee['structure'].get_symbols())

    # print(' structure')
    print('Final energy:', parsed_data['scf_energy'])

    modes = [np.array(m['displacement']) for m in parsed_data['modes']]
    freqs = [m['frequency'] for m in parsed_data['modes']]

    m = 2  # mode number

    sm = SymmetryModes(group='C2v', coordinates=molecule_coor, modes=modes, symbols=molecule_symbols)
    m_measure.append(sm.get_state_mode(m).get_ir_representation().values)
    print('sm', sm)
    for i in range(len(modes)):
        print('m {}:'.format(i + 1), sm.get_state_mode(i))

pg = PointGroup(group='C2v')

for m, l in zip(np.array(m_measure).T, pg.ir_labels):
    plt.plot(np.arange(-1.0, 1.0, 0.02), m, '-', label=l)

plt.plot(np.arange(-1.0, 1.0, 0.02), np.add.reduce(m_measure, axis=1), '-', label='sum')

plt.legend()
plt.show()
