# example of H2O molecule HF calculation with pyscf
from posym import SymmetryMolecule, SymmetryGaussianLinear
from posym.tools import get_basis_set_pyscf, build_density, build_orbital
from pyscf import gto, scf
import numpy as np


r = 1  # O-H distance
alpha = np.deg2rad(104.5)  # H-O-H angle

mol_pyscf = gto.M(atom=[['O', [0, 0, 0]],
                        ['H', [-r, 0, 0]],
                        ['H', [r*np.cos(np.pi - alpha), r*np.sin(np.pi - alpha), 0]]],
                  basis='3-21g',
                  charge=0,
                  spin=0)

# h2o_molecule.build()
pyscf_scf = scf.RHF(mol_pyscf)
pyscf_scf = pyscf_scf.run()

# get data
mo_coefficients = pyscf_scf.mo_coeff.T
overlap_matrix = pyscf_scf.get_ovlp(mol_pyscf)

# print data
print('n_atoms: ', mol_pyscf.natm)
print('n_orbitals: ', mol_pyscf.nao)
print('n_electrons: ', mol_pyscf.nelectron)

geom_sym = SymmetryMolecule('c2v', mol_pyscf.atom_coords(), [mol_pyscf.atom_symbol(i) for i in range(mol_pyscf.natm)])
print('geometry CSM: ', geom_sym.measure)
print('geometry center: ', geom_sym.center)

print('\nmolecule')
for i, c in enumerate(mol_pyscf.atom_coords()):
    print(mol_pyscf.atom_symbol(i), '{:10.5f} {:10.5f} {:10.5f}'.format(*c))

basis_set = get_basis_set_pyscf(mol_pyscf)

print('\nMO symmetry')
for i, orbital_vect in enumerate(mo_coefficients):
    orb = build_orbital(basis_set, orbital_vect)

    sym_orb = SymmetryGaussianLinear('c2v', orb,
                                     orientation_angles=geom_sym.orientation_angles,
                                     center=geom_sym.center
                                     )

    print('orbital {}: {}'.format(i, sym_orb))


# build RHF density (AO basis)
density_matrix_mo = np.diag([2.0]*(mol_pyscf.nelectron//2) + [0.0]*(mol_pyscf.nao-mol_pyscf.nelectron//2))
density_matrix = mo_coefficients.T @ density_matrix_mo @ mo_coefficients


f_density = build_density(basis_set, density_matrix)

print('\ndensity integral: ', f_density.integrate)

sm_dens = SymmetryGaussianLinear('c2v', f_density,
                                 orientation_angles=geom_sym.orientation_angles,
                                 center=geom_sym.center
                                 )

print('density symmetry: ', sm_dens)
print('density CSM: ', sm_dens.measure)
