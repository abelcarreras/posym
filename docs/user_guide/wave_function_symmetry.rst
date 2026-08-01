Wave Function Symmetry
=======================

PoSym supports symmetry determination for both single-determinant and
multi-configurational wave functions.

Single-Determinant Wave Functions
---------------------------------

:class:`~posym.SymmetrySingleDeterminant` computes the symmetry of a
Slater determinant from a set of occupied :class:`~posym.basis.BasisFunction`
objects:

.. code-block:: python

   from posym import SymmetrySingleDeterminant
   from posym.tools import build_orbital

   # Build orbitals from basis set and MO coefficients
   orbitals = [build_orbital(basis_set, coeff)
               for coeff in mo_coefficients]

   wf = SymmetrySingleDeterminant('Td',
                                  alpha_orbitals=[orbitals[0], orbitals[1], orbitals[2]],
                                  beta_orbitals=[orbitals[0], orbitals[1], orbitals[3]])

   print('Configuration:', wf)

The symmetry is computed via the determinant of the orbital overlap
matrices under each symmetry operation:

.. math::

   \det \left[ \langle \phi_i | \hat{R} \phi_j \rangle \right]

for the alpha and beta spin blocks separately, then multiplied.

Multi-Determinant Wave Functions
--------------------------------

:class:`~posym.SymmetryMultiDeterminant` handles CI-type wave functions
defined as linear combinations of Slater determinants:

.. code-block:: python

   from posym import SymmetryMultiDeterminant

   configurations = [
       {'amplitude': -0.03216,
        'occupations': {'alpha': [1, 1, 0, 0, 1],
                        'beta':  [1, 1, 1, 0, 0]}},
       {'amplitude': 0.70637,
        'occupations': {'alpha': [1, 1, 0, 1, 0],
                        'beta':  [1, 1, 1, 0, 0]}},
       # ... more configurations
   ]

   wf_ci = SymmetryMultiDeterminant('Td',
                                    orbitals=orbitals,
                                    configurations=configurations,
                                    center=[0, 0, 0])

   print('CI State:', wf_ci)

The overlap between two Slater determinants is computed as:

.. math::

   \langle D_I | \hat{R} D_J \rangle =
   \det \mathbf{S}_{IJ}^\alpha \times \det \mathbf{S}_{IJ}^\beta

where :math:`\mathbf{S}_{IJ}^\alpha` is the sub-matrix of the orbital
overlap matrix selected by the occupation patterns of determinants
:math:`I` and :math:`J`.
