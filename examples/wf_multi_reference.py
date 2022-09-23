# Analysis of the symmetry of the multi-reference wave functions
# of the excited states of methane molecule computed using RAS-CI method
from pyqchem import get_output_from_qchem, QchemInput, Structure
from pyqchem.parsers.parser_optimization import basic_optimization
from pyqchem.parsers.parser_rasci import parser_rasci
from pyqchem.file_io import write_to_fchk

from posym import SymmetryWaveFunctionCI
from posym.tools import get_basis_set, build_orbital
import posym.algebra as al
import numpy as np


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

write_to_fchk(ee_methane, 'methane.fchk')
overlap = np.round(np.array(ee_methane['overlap']), decimals=6)

coefficients = ee_methane['coefficients']
coordinates = ee_methane['structure'].get_coordinates()
basis = ee_methane['basis']

basis_set = get_basis_set(coordinates, basis)
orbitals = []
for orbital_coefficients in coefficients['alpha']:
    orbitals.append(build_orbital(basis_set, orbital_coefficients))

for istate, state in enumerate(data_methane['excited_states']):
    print('\nState', istate + 1, '(', state['multiplicity'], ')')
    print('Excitation energy: {:8.2f}'.format(state['excitation_energy']))

    for configuration in state['configurations']:
        print('amplitude: {:12.8f} '.format(configuration['amplitude']), configuration['occupations'])

    wf = SymmetryWaveFunctionCI('Td',
                                orbitals=orbitals,
                                configurations=state['configurations'],
                                center=[0, 0, 0])

    print(wf.get_ir_representation())
    print(wf)
    print('Norm: ', al.norm(wf))
