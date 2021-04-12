from pyqchem import get_output_from_qchem, Structure, QchemInput
from pyqchem.parsers.parser_frequencies import basic_frequencies
from posym import PointGroup, SymmetryModes
import numpy as np


water = [[ 0.00000000,  0.00000000e+00,  2.40297090e-01],
         [-1.43261539,  0.00000000e+00, -9.61188362e-01],
         [ 1.43261539,  0.00000000e+00, -9.61188362e-01]]
water = np.array(water) * 0.5


molecule_water = Structure(coordinates=water,
                           symbols=['O', 'H', 'H'],
                           charge=0,
                           multiplicity=1)

dicloro = [[ 2.1437,  0.1015, -0.0002],
           [-2.1439, -0.1011, -0.0002],
           [ 0.5135 , -0.4232, 0.0002],
           [-0.5132,  0.4227, 0.0002],
           [ 0.4242, -1.5014, 0.0001],
           [-0.4237,  1.5009, 0.0001]]
dcl_symbols = ['Cl', 'Cl', 'C', 'C', 'H', 'H']

molecule_ch2cl2 = Structure(coordinates=dicloro,
                            symbols=dcl_symbols,
                            charge=0,
                            multiplicity=1)

for molecule, group in zip([molecule_water, molecule_ch2cl2], ['c2v', 'c2h']):
    qc_input = QchemInput(molecule,
                          jobtype='freq',
                          exchange='hf',
                          basis='6-31G',
                          # sym_ignore=True,
                          )

    parsed_data, ee = get_output_from_qchem(qc_input, parser=basic_frequencies, read_fchk=True)

    molecule_coor = np.array(ee['structure'].get_coordinates())
    molecule_symbols = np.array(ee['structure'].get_symbols())

    # print(' structure')
    print('\nPoint Group: {}'.format(group))
    print('Final energy:', parsed_data['scf_energy'])

    modes = [np.array(m['displacement']) for m in parsed_data['modes']]
    freqs = [m['frequency'] for m in parsed_data['modes']]
    print('freqs: ', freqs)

    sm = SymmetryModes(group=group, coordinates=molecule_coor, modes=modes, symbols=molecule_symbols, optimize=False)

    for i in range(len(modes)):
        print('m {}:'.format(i + 1), sm.get_state_mode(i))
    print('Total: ', sm)
