from pyqchem import get_output_from_qchem, Structure, QchemInput
from pyqchem.parsers.parser_frequencies import basic_frequencies
from pyqchem.parsers.parser_optimization import basic_optimization

from posym import SymmetryModes
from pyqchem.tools import get_geometry_from_pubchem
import numpy as np
import posym.algebra as al


sh6_coor = [[ 0.16290727, -0.36340852,  0.00000000],
            [ 1.47290727, -0.36340852,  0.00000000],
            [ 0.16290727,  0.94659148,  0.00000000],
            [ 0.16290727, -0.36340852,  1.31000000],
            [-1.14709273, -0.36340852,  0.00000000],
            [ 0.16290727, -1.67340852,  0.00000000],
            [ 0.16290727, -0.36340852, -1.31000000]]
sh6_sym = ['S', 'H', 'H', 'H', 'H', 'H', 'H']

sh6_mol = Structure(coordinates=sh6_coor,
                    symbols=sh6_sym,
                    charge=0,
                    multiplicity=1)

dicloro_coor = [[ 2.1437,  0.1015, -0.0002],
                [-2.1439, -0.1011, -0.0002],
                [ 0.5135, -0.4232,  0.0002],
                [-0.5132,  0.4227,  0.0002],
                [ 0.4242, -1.5014,  0.0001],
                [-0.4237,  1.5009,  0.0001]]
dcl_symbols = ['Cl', 'Cl', 'C', 'C', 'H', 'H']

# from scipy.spatial.transform import Rotation as R
# angles = [100, 200, 300]
# rotmol = R.from_euler('zyx', angles, degrees=True)
# dicloro_coor = rotmol.apply(dicloro_coor)

dichloro_mol = Structure(coordinates=dicloro_coor,
                         symbols=dcl_symbols,
                         charge=0,
                         multiplicity=1)

methane_mol = get_geometry_from_pubchem('methane')
ammonia_mol = get_geometry_from_pubchem('ammonia')
water_mol = get_geometry_from_pubchem('water')
h2o2_mol = get_geometry_from_pubchem('h2o2')


for molecule, group in zip([h2o2_mol, sh6_mol, water_mol, dichloro_mol, methane_mol, ammonia_mol],
                           ['c2',      'Oh',     'c2v',       'c2h',       'Td',        'c3v']):

    qc_input = QchemInput(molecule,
                          jobtype='opt',
                          exchange='hf',
                          basis='sto-3g',
                          #geom_opt_tol_gradient=1,
                          #geom_opt_tol_energy=1,
                          #geom_opt_max_cycles=500,
                          )

    parsed_data = get_output_from_qchem(qc_input, parser=basic_optimization, processors=6)

    qc_input = QchemInput(parsed_data['optimized_molecule'],
                          jobtype='freq',
                          exchange='hf',
                          basis='sto-3g',
                          #sym_ignore=True,
                          )

    print(molecule)
    parsed_data, ee = get_output_from_qchem(qc_input, parser=basic_frequencies,
                                            return_electronic_structure=True, processors=6)

    molecule_coor = np.array(ee['structure'].get_coordinates())
    molecule_symbols = np.array(ee['structure'].get_symbols())

    # print(' structure')
    print('\nPoint Group: {}'.format(group))
    print('Final energy:', parsed_data['scf_energy'])

    modes = [np.array(m['displacement']) for m in parsed_data['modes']]
    freqs = [m['frequency'] for m in parsed_data['modes']]
    # print('freqs: ', freqs)
    # print(molecule_coor)

    sm = SymmetryModes(group=group, coordinates=molecule_coor, modes=modes, symbols=molecule_symbols)

    for i in range(len(modes)):
        print('m {:2}: {:8.3f} :'.format(i + 1, freqs[i]), sm.get_state_mode(i))
    print('Total: ', sm)
    print('Dot: ', al.dot(sm, sm))
    print('angles: ', sm.orientation_angles)
