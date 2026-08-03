# Example script that analyzes the continuous symmetry of
# the normal modes along the distortion of the H2O molecule
# This example requires PyQchem to compute the normal modes

from pyqchem import get_output_from_qchem, Structure, QchemInput
from pyqchem.parsers.parser_frequencies import basic_frequencies
from pyqchem.tools.duschinsky import NormalModes
from posym import PointGroup, SymmetryNormalModes, SymmetryNormalModesProjection
import matplotlib.pyplot as plt
import numpy as np


def get_mass_weighted_modes(modes):
    m = NormalModes(ee['structure'], modes, freqs, is_mass_weighted=False)
    modes = m.get_displacements()
    modes = modes.transpose(1, 0)
    modes = modes.reshape(len(modes), -1, 3)
    return modes

measures_m0 = []
measures_m1 = []
measures_m2 = []

csm_m0 = []
csm_m1 = []
csm_m2 = []

energies = []
frequencies = []

# scan_range = np.arange(-0.2, 0.21, 0.01)
scan_range = np.arange(-0.02, 0.021, 0.001)

#scan_range = [-0.1, 0.0, 0.1]

for x_dist in scan_range:

    water = [[x_dist,           0.00000000e+00,  2.40297090e-01],
             [-1.43261539e+00, -1.75444785e-16, -9.61188362e-01],
             [1.43261539e+00,   1.75444785e-16, -9.61188362e-01]]
    water = np.array(water) * 0.529177249

    molecule_water = Structure(coordinates=water,
                               symbols=['O', 'H', 'H'],
                               charge=0,
                               multiplicity=1)

    qc_input = QchemInput(molecule_water,
                          jobtype='freq',
                          exchange='hf',
                          basis='6-31G',
                          sym_ignore=True,
                          )

    parsed_data, ee = get_output_from_qchem(qc_input, parser=basic_frequencies, return_electronic_structure=True)

    molecule_coor = np.array(ee['structure'].get_coordinates())
    molecule_symbols = np.array(ee['structure'].get_symbols())

    # print(' structure')
    print('Final energy:', parsed_data['scf_energy'])
    energies.append(parsed_data['scf_energy'])
    modes = [np.array(m['displacement']) for m in parsed_data['modes']]
    freqs = [m['frequency'] for m in parsed_data['modes']]
    print('freqs: ', freqs)
    frequencies.append(freqs)

    modes = get_mass_weighted_modes(modes)

    sm = SymmetryNormalModes(group='c2v', coordinates=molecule_coor, modes=modes, symbols=molecule_symbols)
    sm_proj = SymmetryNormalModesProjection(group='c2v', coordinates=molecule_coor, modes=modes, symbols=molecule_symbols,
                                            orientation_angles=sm.orientation_angles)

    measures_m0.append(sm_proj.get_state_mode_proj(0).get_ir_representation().values)
    measures_m1.append(sm_proj.get_state_mode_proj(1).get_ir_representation().values)
    measures_m2.append(sm_proj.get_state_mode_proj(2).get_ir_representation().values)

    csm_m0.append(sm_proj.get_state_mode_proj(0).measure)
    csm_m1.append(sm_proj.get_state_mode_proj(1).measure)
    csm_m2.append(sm_proj.get_state_mode_proj(2).measure)

    print('Sym: ', sm)
    print('Sym proj: ', sm_proj)
    print('CSM pos', sm.measure_pos)
    print('CSM modes', sm_proj.measure)

    for i in range(len(modes)):
        print('m {}:'.format(i + 1), sm.get_state_mode(i))
        print('   mode CSM:', sm_proj.get_state_mode_proj(i).measure)
    print('----------------------\n')


pg = PointGroup(group='C2v')

plt.title('mode 1')
for ir, l in zip(np.array(measures_m0).T, pg.ir_labels):
    plt.plot(scan_range, ir, '-', label=l)
plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylim(0, 1)

plt.figure()
plt.title('mode 2')
for ir, l in zip(np.array(measures_m1).T, pg.ir_labels):
    plt.plot(scan_range, ir, '-', label=l)
plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylim(0, 1)

plt.figure()
plt.title('mode 3')
for ir, l in zip(np.array(measures_m2).T, pg.ir_labels):
    plt.plot(scan_range, ir, '-', label=l)
plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylim(0, 1)

plt.figure()
plt.title('Frequencies')
for i, freq in enumerate(np.array(frequencies).T):
    plt.plot(scan_range, freq, '-', label='mode: {}'.format(i+1))
plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylabel('Frequency (cm-1)')

plt.figure()
plt.title('CSM')
plt.plot(scan_range, csm_m0, '-', label='csm_1')
plt.plot(scan_range, csm_m1, '-', label='csm_2')
plt.plot(scan_range, csm_m2, '-', label='csm_3')

plt.legend()
plt.xlabel('Distortion (Bohr)')
plt.ylim(0, 100)

plt.figure()
plt.title('energy')
plt.plot(scan_range, energies, '-')
plt.ylabel('Hartree')
plt.xlabel('Bohr')

plt.show()



