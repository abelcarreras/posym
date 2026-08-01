Molecular Orbital Symmetry
===========================

The :class:`~posym.SymmetryGaussianLinear` class determines the symmetry
representation of a function expressed in a Gaussian basis set.

Basic Usage
-----------

.. code-block:: python

   from posym import SymmetryGaussianLinear

   # orbital is a BasisFunction representing a molecular orbital
   sym_orb = SymmetryGaussianLinear('C2v', orbital)

   print('Symmetry:', sym_orb)

How It Works
------------

The symmetry determination is based on the overlap between the function
and its image under each symmetry operation. For a function :math:`\psi`
and a symmetry operation :math:`\hat{R}`:

.. math::

   \chi_{\psi}(R) =
   \frac{\langle \psi | \hat{R} \psi \rangle}
        {\langle \psi | \psi \rangle}

The set of these overlaps forms the character vector for the function in the symmetry
operations basis, which is then transformed into the irreducible representation basis of
the point group. See :ref:`transform-matrices` for further details.


Useful applications
-------------------

The symmetry of multiple molecular orbitals in a calculation can be obtained:

.. code-block:: python

   mo_symmetries = [SymmetryGaussianLinear('C2v', mo)
                    for mo in molecular_orbitals]

   for i, sym in enumerate(mo_symmetries):
       print(f'MO {i+1}: {sym}')

The product of occupied orbital symmetries gives the total electronic
state symmetry (for non-degenerate wave functions):

.. code-block:: python

   # Doubly occupied: multiply each orbital twice
   wf_sym = mo_symmetries[0] * mo_symmetries[0]  # ...
