# Example of calculation of the symmetry of normal modes
# using PyQchem to automize its calculations

from pyqchem import get_output_from_qchem, Structure, QchemInput
from pyqchem.parsers.parser_frequencies import basic_frequencies
from pyqchem.parsers.parser_optimization import basic_optimization

from posym import SymmetryNormalModes, SymmetryAtomDisplacements, SymmetryNormalModesProjection
from pyqchem.tools import get_geometry_from_pubchem
import numpy as np
import posym.algebra as al

def get_mass_weighted_modes(modes):
    """ transform modes from cartessian to mass-weighted coordinates """
    from pyqchem.tools.duschinsky import NormalModes
    freqs = np.zeros_like(modes)
    m = NormalModes(ee['structure'], modes, freqs, is_mass_weighted=False)
    modes = m.get_displacements()
    modes = modes.transpose(1, 0)
    modes = modes.reshape(len(modes), -1, 3)
    return modes

molecule = get_geometry_from_pubchem('ammonia')
group = 'c3v'

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
# hessian = np.array(ee['hessian'])

# print(' structure')
print('\nPoint Group: {}'.format(group))
print('Final energy:', parsed_data['scf_energy'])

modes = [np.array(m['displacement']) for m in parsed_data['modes']]
modes = get_mass_weighted_modes(modes) # convert to mass-weighted coordinates
freqs = [m['frequency'] for m in parsed_data['modes']]
# print('freqs: ', freqs)
# print(molecule_coor)
sm = SymmetryNormalModes(group=group, coordinates=molecule_coor, modes=modes, symbols=molecule_symbols)
sm_proj = SymmetryNormalModesProjection(group=group, coordinates=molecule_coor, modes=modes, symbols=molecule_symbols)

print('\nMode symmetry:')
for i in range(len(modes)):
    print('m {:2}: {:8.3f} :'.format(i + 1, freqs[i]), sm.get_state_mode(i))


print('\nMode projected subspace [CSM]:')
sm_1 = SymmetryNormalModesProjection(group=group, coordinates=molecule_coor, modes=modes[0:1], symbols=molecule_symbols)
print('m 1  : {:8.3f} [{:.3f}] : '.format(freqs[0], sm_1.measure), sm_1)

sm_23 = SymmetryNormalModesProjection(group=group, coordinates=molecule_coor, modes=modes[1:3], symbols=molecule_symbols)
print('m 2-3: {:8.3f} [{:.3f}] : '.format(freqs[0], sm_23.measure), sm_23)

sm_4 = SymmetryNormalModesProjection(group=group, coordinates=molecule_coor, modes=modes[3:4], symbols=molecule_symbols)
print('m 4  : {:8.3f} [{:.3f}] : '.format(freqs[0], sm_4.measure), sm_4)

sm_56 = SymmetryNormalModesProjection(group=group, coordinates=molecule_coor, modes=modes[4:6], symbols=molecule_symbols)
print('m 5-6: {:8.3f} [{:.3f}] : '.format(freqs[0], sm_23.measure), sm_23)

print('\nTotal modes: ', sm)
print('Total projection: ', sm_proj)

print('Dot: ', al.dot(sm, sm))
print('angles: ', sm.orientation_angles)
print('csm_pos: ', sm.measure_pos)
print('csm proj: ', sm_proj.measure)

sm_xyz = SymmetryAtomDisplacements(group=group, coordinates=molecule_coor, symbols=molecule_symbols)
print('modes xyz: ', sm_xyz)
