# Example of methane excited states constructed by CI of
# a set of HF restricted molecular orbitals

from pyqchem import get_output_from_qchem, QchemInput, Structure
from pyqchem.parsers.parser_optimization import basic_optimization
from pyqchem.parsers.parser_rasci import parser_rasci
from pyqchem.file_io import write_to_fchk

from posym import SymmetryGaussianLinear, SymmetrySingleDeterminant
import numpy as np
from posym.tools import get_basis_set, build_orbital

# define molecular structure
methane = Structure(coordinates=[[ 0.0000000000,  0.0000000000,  0.0000000000],
                                 [ 0.5541000000,  0.7996000000,  0.4965000000],
                                 [ 0.6833000000, -0.8134000000, -0.2536000000],
                                 [-0.7782000000, -0.3735000000,  0.6692000000],
                                 [-0.4593000000,  0.3874000000, -0.9121000000]],
                    symbols=['C', 'H', 'H', 'H', 'H'])

# optimize geometry using HF
qc_input = QchemInput(methane,
                      jobtype='opt',
                      exchange='hf',
                      basis='sto-3g',
                      geom_opt_tol_gradient=1,
                      geom_opt_tol_displacement=1,
                      geom_opt_tol_energy=1,
                      geom_opt_max_cycles=200,
                      )

data_methane = get_output_from_qchem(qc_input,
                                     processors=4,
                                     parser=basic_optimization,
                                     )

# Use RAS-CI method to calculate the a CI wave function
qc_input = QchemInput(data_methane['optimized_molecule'],
                      jobtype='sp',
                      exchange='hf',
                      basis='sto-3g',
                      correlation='rasci',
                      purecart=False,
                      ras_roots=10,
                      ras_elec_alpha=1,
                      ras_elec_beta=1,
                      ras_act=3,
                      ras_occ=2,
                      ras_do_hole=False,
                      ras_do_part=False,
                      cis_convergence=15,
                      sym_tol=2,
                      ras_spin_mult=0,
                      )

data_methane, ee_methane = get_output_from_qchem(qc_input,
                                                 return_electronic_structure=True,
                                                 processors=4,
                                                 force_recalculation=True,
                                                 parser=parser_rasci
                                                 )

# store fchk to visualize orbitals
write_to_fchk(ee_methane, 'methane.fchk')

# read electronic structure information
# overlap = np.round(np.array(ee_methane['overlap']), decimals=6)
coefficients = ee_methane['coefficients']
coordinates = np.array(ee_methane['structure'].get_coordinates())
basis = ee_methane['basis']

# print excited states configurations
for istate, state in enumerate(data_methane['excited_states']):
    print('\nState', istate +1, '(', state['multiplicity'], ')')
    print('excitation_energy: ', state['excitation_energy'])
    for configuration in state['configurations']:
        print(configuration['amplitude'], configuration['occupations'])

# build molecular orbitals
print('\nMolecular orbitals symmetry')
basis_set = get_basis_set(coordinates, basis)
orbital1 = build_orbital(basis_set, coefficients['alpha'][0])
orbital2 = build_orbital(basis_set, coefficients['alpha'][1])

orbital3 = build_orbital(basis_set, coefficients['alpha'][2])
orbital4 = build_orbital(basis_set, coefficients['alpha'][3])
orbital5 = build_orbital(basis_set, coefficients['alpha'][4])

# compute symmetry of molecular orbitals
sym_orbital1 = SymmetryGaussianLinear('Td', orbital1)
sym_orbital2 = SymmetryGaussianLinear('Td', orbital2)
sym_orbital3 = SymmetryGaussianLinear('Td', orbital3)
sym_orbital4 = SymmetryGaussianLinear('Td', orbital4)
sym_orbital5 = SymmetryGaussianLinear('Td', orbital5)

for i, s in enumerate([sym_orbital1, sym_orbital2, sym_orbital3, sym_orbital4, sym_orbital5]):
    print('Orbital {}: {}'.format(i+1, s))

print('\nWave function symmetry')
# compute symmetry of sample configuration (in T states)
wf_1 = SymmetrySingleDeterminant('Td',
                                 alpha_orbitals=[orbital1, orbital2, orbital5],
                                 beta_orbitals=[orbital1, orbital2, orbital4])

print('Configuration 1: ', wf_1)

# compute symmetry of sample configuration (in A & E states)
wf_2 = SymmetrySingleDeterminant('Td',
                                 alpha_orbitals=[orbital1, orbital2, orbital3],
                                 beta_orbitals=[orbital1, orbital2, orbital3])

print('Configuration 2: ', wf_2)

