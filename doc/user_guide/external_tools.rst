External Tool Integration
==========================

PoSym provides utilities to interface with external electronic structure
packages.

PySCF Integration
-----------------

Complete workflow from PySCF calculation to PoSym symmetry analysis:

.. code-block:: python

   from posym import SymmetryGaussianLinear
   from posym.tools import get_basis_set_pyscf, build_orbital
   from pyscf import gto, scf

   mol = gto.M(atom=[['O', [0, 0, 0]],
                     ['H', [-0.757, 0.586, 0]],
                     ['H', [0.757, 0.586, 0]]],
               basis='3-21g')

   mf = scf.RHF(mol).run()

   basis_set = get_basis_set_pyscf(mol)
   mo_coeffs = mf.mo_coeff.T

   for i, coeff in enumerate(mo_coeffs):
       orb = build_orbital(basis_set, coeff)
       sym = SymmetryGaussianLinear('c2v', orb)
       print(f'MO {i}: {sym}')

The :func:`~posym.tools.get_basis_set_pyscf` function converts a PySCF
molecule object into PoSym :class:`~posym.basis.BasisFunction` objects,
supporting s, p, and d shells.

PyQchem Integration
-------------------

A similar workflow using PyQchem (Q-Chem interface):

.. code-block:: python

   from posym.tools import get_basis_set, build_orbital
   from pyqchem import get_output_from_qchem, QchemInput, Structure

   mol = Structure(coordinates=coords, symbols=symbols)
   qc_input = QchemInput(mol, jobtype='sp', exchange='hf', basis='sto-3g')

   data, ee = get_output_from_qchem(qc_input, read_fchk=True, parser=parser)

   basis_set = get_basis_set(ee['structure'].get_coordinates(), ee['basis'])
   orbital = build_orbital(basis_set, ee['coefficients']['alpha'][0])

   sym = SymmetryGaussianLinear('c2v', orbital)

The :func:`~posym.tools.get_basis_set` function handles the basis set
dictionary format produced by PyQchem and supports s, sp, p, d, and f
shells.
