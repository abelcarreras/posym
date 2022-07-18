# Example of methane excited states

from pyqchem import get_output_from_qchem, QchemInput, Structure
from pyqchem.parsers.parser_optimization import basic_optimization
from pyqchem.parsers.parser_rasci import parser_rasci
from pyqchem.file_io import write_to_fchk

from posym import SymmetryFunction, SymmetryBase, SymmetryWaveFunction
import posym.algebra as al
import numpy as np
from posym.tools import get_basis_set, build_orbital


methane = Structure(coordinates=[[ 0.0000000000,  0.0000000000,  0.0000000000],
                                 [ 0.5541000000,  0.7996000000,  0.4965000000],
                                 [ 0.6833000000, -0.8134000000, -0.2536000000],
                                 [-0.7782000000, -0.3735000000,  0.6692000000],
                                 [-0.4593000000,  0.3874000000, -0.9121000000]],
                    symbols=['C', 'H', 'H', 'H', 'H'])


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

write_to_fchk(ee_methane, 'methane.fchk')

overlap = np.round(np.array(ee_methane['overlap']), decimals=6)

coefficients = ee_methane['coefficients']
coordinates = np.array(ee_methane['structure'].get_coordinates())
basis = ee_methane['basis']


for istate, state in enumerate(data_methane['excited_states']):
    print('\nState', istate +1, '(', state['multiplicity'], ')')
    print('excitation_energy: ', state['excitation_energy'])
    for configuration in state['configurations']:
        print(configuration['amplitude'], configuration['occupations'])

print('\nMolecular orbitals symmetry')
basis_set = get_basis_set(coordinates, basis)
orbital1 = build_orbital(basis_set, coefficients['alpha'][0])
orbital2 = build_orbital(basis_set, coefficients['alpha'][1])

orbital3 = build_orbital(basis_set, coefficients['alpha'][2])
orbital4 = build_orbital(basis_set, coefficients['alpha'][3])
orbital5 = build_orbital(basis_set, coefficients['alpha'][4])

sym_orbital1 = SymmetryFunction('Td', orbital1)
sym_orbital2 = SymmetryFunction('Td', orbital2)
sym_orbital3 = SymmetryFunction('Td', orbital3)
sym_orbital4 = SymmetryFunction('Td', orbital4)
sym_orbital5 = SymmetryFunction('Td', orbital5)

for i, s in enumerate([sym_orbital1, sym_orbital2, sym_orbital3, sym_orbital4, sym_orbital5]):
    print('Orbital {}: {}'.format(i+1, s))

print('\nWave function symmetry')

wf = SymmetryWaveFunction('Td',
                          alpha_orbitals=[orbital1, orbital2, orbital5],
                          beta_orbitals=[orbital1, orbital2, orbital4],
                          center=[0, 0, 0])

print('Configuration 1: ', wf)


wf = SymmetryWaveFunction('Td',
                          alpha_orbitals=[orbital1, orbital2, orbital3],
                          beta_orbitals=[orbital1, orbital2, orbital3],
                          center=[0, 0, 0])

print('Configuration 2: ', wf)

