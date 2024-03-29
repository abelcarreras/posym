# Analysis of the symmetry of the multi-reference wave functions
# of the excited states of methane molecule computed using RAS-CI method
# This example requires PyQchem to do the electronic structure calculations

from pyqchem import get_output_from_qchem, QchemInput, Structure
from pyqchem.parsers.parser_optimization import basic_optimization
from pyqchem.parsers.parser_rasci import parser_rasci
from pyqchem.file_io import write_to_fchk
from posym import SymmetryMultiDeterminant, SymmetryGaussianLinear
from posym.tools import get_basis_set, build_orbital
import posym.algebra as al

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
                      ras_roots=12,
                      ras_elec_alpha=1,
                      ras_elec_beta=1,
                      ras_act=3,
                      ras_occ=2,
                      ras_do_hole=False,
                      ras_do_part=False,
                      cis_convergence=10,
                      sym_tol=2,
                      ras_spin_mult=0,
                      extra_rem_keywords={'set_maxsize': 600,
                                          'RAS_AMPL_PRINT': 2},
                      set_iter=100,
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
coefficients = ee_methane['coefficients']
coordinates = ee_methane['structure'].get_coordinates()
basis = ee_methane['basis']

basis_set = get_basis_set(coordinates, basis)

# build molecular orbitals
print('Molecular Orbitals')
for i, mo_coeff in enumerate(ee_methane['coefficients']['alpha']):
    mo_orbital = build_orbital(basis_set, mo_coeff)
    print(i + 1, ':', SymmetryGaussianLinear('Td', mo_orbital))

orbitals = []
for orbital_coefficients in coefficients['alpha']:
    orbitals.append(build_orbital(basis_set, orbital_coefficients))

# print excited states configurations and compute the symmetry of the full multi-configurational wave function
for istate, state in enumerate(data_methane['excited_states']):
    print('\nState', istate + 1, '(', state['multiplicity'], ')')
    print('Excitation energy: {:8.4f}'.format(state['excitation_energy']))

    for configuration in state['configurations']:
        print('amplitude: {:12.8f} '.format(configuration['amplitude']), configuration['occupations'])

    wf = SymmetryMultiDeterminant('Td',
                                  orbitals=orbitals,
                                  configurations=state['configurations'],
                                  center=[0, 0, 0])

    print(wf.get_ir_representation())
    print(wf)
    print('Norm: ', al.norm(wf))
