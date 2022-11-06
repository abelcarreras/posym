# Example of a simple wave function analysis of the butadiene molecule
# This example makes use of pyQChem to automatize the calculation of the wave function
from pyqchem import get_output_from_qchem, QchemInput, Structure
from pyqchem.parsers.basic import basic_parser_qchem
from posym import SymmetryFunction, SymmetryBase
from posym.tools import get_basis_set, build_orbital
import posym.algebra as al
import numpy as np
import matplotlib.pyplot as plt


# define structure of cis-butadinen
molecule_cis = Structure(coordinates=[[ -1.07076839,   -2.13175980,    0.03234382],
                                      [ -0.53741536,   -3.05918866,    0.04995793],
                                      [ -2.14073783,   -2.12969357,    0.04016267],
                                      [ -0.39112115,   -0.95974916,    0.00012984],
                                      [  0.67884827,   -0.96181542,   -0.00769025],
                                      [ -1.15875076,    0.37505495,   -0.02522296],
                                      [ -0.62213437,    1.30041753,   -0.05065831],
                                      [ -2.51391203,    0.37767199,   -0.01531698],
                                      [ -3.04726506,    1.30510083,   -0.03293196],
                                      [ -3.05052841,   -0.54769055,    0.01011971]],
                         symbols=['C', 'H', 'H', 'C', 'H', 'C', 'H', 'C', 'H', 'H'])


# create Q-Chem input
qc_input = QchemInput(molecule_cis,
                      jobtype='sp',
                      exchange='hf',
                      basis='sto-3g',
                      #sym_ignore=True
                      )

# Calculate and parse the output
data_cis, ee_cis = get_output_from_qchem(qc_input,
                                         return_electronic_structure=True,
                                         processors=4,
                                         parser=basic_parser_qchem)

# define structure of trans-butadinen
molecule_trans = Structure(coordinates=[[ -1.06908233,   -2.13352097,   -0.00725330],
                                        [ -0.53502155,   -3.05996561,   -0.04439369],
                                        [ -2.13778918,   -2.13379901,    0.04533562],
                                        [ -0.39193053,   -0.95978774,   -0.02681816],
                                        [  0.67677629,   -0.95950970,   -0.07940766],
                                        [ -1.16057856,    0.37359983,    0.02664018],
                                        [ -2.22928521,    0.37332175,    0.07923299],
                                        [ -0.48342683,    1.54733308,    0.00707382],
                                        [ -1.01748758,    2.47377771,    0.04421474],
                                        [  0.58527987,    1.54761115,   -0.04551805]],
                           symbols=['C', 'H', 'H', 'C', 'H', 'C', 'H', 'C', 'H', 'H'])



# create Q-Chem input
qc_input = QchemInput(molecule_trans,
                      jobtype='sp',
                      exchange='hf',
                      basis='sto-3g',
                      sym_ignore=True
                      )

# calculate and parse output
data_trans, ee_trans = get_output_from_qchem(qc_input,
                                             return_electronic_structure=True,
                                             processors=4,
                                             parser=basic_parser_qchem,
                                             )

# read electronic structure data
coordinates_cis = ee_cis['structure'].get_coordinates()
coordinates_trans = ee_trans['structure'].get_coordinates()

coefficients_cis = ee_cis['coefficients']
coefficients_trans = ee_trans['coefficients']

# extract the basis set info
basis_set_trans = get_basis_set(coordinates_trans, ee_trans['basis'])
basis_set_cis = get_basis_set(coordinates_cis, ee_trans['basis'])

# build 16th molecular orbital for test using PoSym helper functions
o15_cis = build_orbital(basis_set_cis, coefficients_cis['alpha'][15])
o15_trans = build_orbital(basis_set_trans, coefficients_trans['alpha'][15])

# plot orbitals (contour) using matplotlib
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)

X, Y = np.meshgrid(x, y)

Z = o15_cis(X, Y, np.zeros_like(X))
plt.imshow(Z, interpolation='bilinear', origin='lower', cmap='seismic')
plt.figure()
plt.contour(X, Y, Z, colors='k')
plt.show()


Z = o15_trans(X, Y, np.zeros_like(X))
plt.imshow(Z, interpolation='bilinear', origin='lower', cmap='seismic')
plt.figure()
plt.contour(X, Y, Z, colors='k')
plt.show()


def get_simple_wf_symm(orbitals_symm, alpha=(1,), beta=(1,)):
    """
    generate simple wave function symmetry by product of molecular orbitals symmetry
    :param orbitals_symm: list of orbital symmetries
    :param alpha: alpha orbitals occupation
    :param beta: beta orbitals occupation
    :return: symmetry
    """
    total_sym = None
    for a, o in zip(alpha, orbitals_symm):
        if a == 1:
            total_sym = total_sym * o if total_sym is not None else o

    for b, o in zip(beta, orbitals_symm):
        if b == 1:
            total_sym = total_sym * o if total_sym is not None else o

    return total_sym


# compute and list symmetry of molecular orbitals (CIS molecule)
print('\nCIS\n----')
cis_orbitals_sym = []
for i, orbital_coeff in enumerate(coefficients_cis['alpha']):
    orbital = build_orbital(basis_set_cis, orbital_coeff)
    sym_orbital = SymmetryFunction('c2v', orbital)
    print('Symmetry O{}: '.format(i+1), sym_orbital)
    cis_orbitals_sym.append(sym_orbital)


# construct symmetry of wave functions of example (simple) excited states
cis_wf_0 = get_simple_wf_symm(cis_orbitals_sym,
                              alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                              beta=[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
cis_wf_1 = get_simple_wf_symm(cis_orbitals_sym,
                              alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                              beta=[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])
cis_wf_2 = get_simple_wf_symm(cis_orbitals_sym,
                              alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                              beta=[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])

# generate symmetry of dipole moment operator for C2v group
cis_dm = SymmetryBase(group='C2v', rep='B1') + \
         SymmetryBase(group='C2v', rep='B2') + \
         SymmetryBase(group='C2v', rep='A1')


def check_transition(transtion):
    if al.dot(transtion, SymmetryBase(group='C2v', rep='A1')) > 0.1:
        return 'Allowed'
    else:
        return 'Forbidden'

# check the symmetry of electronic transition and check if allowed according to symmetry rules
print('wf_0 -> wf_1: ', cis_wf_0 * cis_dm * cis_wf_1, check_transition(cis_wf_0 * cis_dm * cis_wf_1))
print('wf_0 -> wf_2: ', cis_wf_0 * cis_dm * cis_wf_2, check_transition(cis_wf_0 * cis_dm * cis_wf_2))

# compute and list symmetry of molecular orbitals (TRANS molecule)
print('\nTRANS\n----')
trans_orbitals_sym = []
for i, orbital_coeff in enumerate(coefficients_trans['alpha']):
    orbital = build_orbital(basis_set_trans, orbital_coeff)
    sym_orbital = SymmetryFunction('c2h', orbital)
    print('Symmetry O{}: '.format(i+1), sym_orbital)
    trans_orbitals_sym.append(sym_orbital)

# construct symmetry of wave functions of example (simple) excited states
trans_wf_0 = get_simple_wf_symm(trans_orbitals_sym,
                                alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                beta=[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
trans_wf_1 = get_simple_wf_symm(trans_orbitals_sym,
                                alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                beta=[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])
trans_wf_2 = get_simple_wf_symm(trans_orbitals_sym,
                                alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                                beta=[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])

# generate symmetry of dipole moment operator for C2h group
trans_dm = SymmetryBase(group='C2h', rep='Bu') + \
           SymmetryBase(group='C2h', rep='Au')


def check_transition(transition):
    if al.dot(transition, SymmetryBase(group='C2h', rep='Ag')) > 0.1:
        return 'Allowed'
    else:
        return 'Forbidden'

# check the symmetry of electronic transition and check if allowed according to symmetry rules
print('wf_0 -> wf_1: ', trans_wf_0 * trans_dm * trans_wf_1, check_transition(trans_wf_0 * trans_dm * trans_wf_1))
print('wf_0 -> wf_2: ', trans_wf_0 * trans_dm * trans_wf_2, check_transition(trans_wf_0 * trans_dm * trans_wf_2))
