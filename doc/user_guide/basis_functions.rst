Gaussian Basis Functions
=========================

PoSym provides a full implementation of Cartesian Gaussian-type orbital
(GTO) basis functions through :class:`~posym.basis.PrimitiveGaussian` and
:class:`~posym.basis.BasisFunction`.

Primitive Gaussians
-------------------

.. code-block:: python

   from posym.basis import PrimitiveGaussian

   # s-type Gaussian centered at origin
   g_s = PrimitiveGaussian(alpha=1.0)

   # p_x Gaussian
   g_px = PrimitiveGaussian(alpha=1.0, l=[1, 0, 0])

   # Evaluate at a point
   value = g_s(0.5, 0.0, 0.0)

   # Product of two Gaussians
   g_prod = g_s * g_px

   # Integrate over all space
   overlap = g_s.integrate

The :class:`PrimitiveGaussian` supports:

- Arbitrary angular momentum ``l = [lx, ly, lz]``
- Automatic normalization when ``normalize=True``
- Translation, rotation, and arbitrary linear transformations
- Analytical products and integrals

Contracted Basis Functions
--------------------------

.. code-block:: python

   from posym.basis import PrimitiveGaussian, BasisFunction

   # STO-3G s-type contraction
   sa = PrimitiveGaussian(alpha=16.1196, l=[0, 0, 0])
   sb = PrimitiveGaussian(alpha=2.9362, l=[0, 0, 0])
   sc = PrimitiveGaussian(alpha=0.7947, l=[0, 0, 0])

   s_orbital = BasisFunction([sa, sb, sc],
                             [0.1543, 0.5353, 0.4446],
                             center=[0.0, 0.0, 0.0])

   # Evaluate
   val = s_orbital(0.0, 0.0, 0.0)

   # Self overlap
   S = (s_orbital * s_orbital).integrate

Linear combinations:

.. code-block:: python

   px = BasisFunction([pxa, pxb, pxc],
                      [0.1559, 0.6077, 0.3920])

   # Add/subtract basis functions
   combined = 0.5 * s_orbital + px

   # Apply transformations
   s_orbital.apply_translation([1.0, 0.0, 0.0])
   px.apply_rotation(1.5708, [0, 0, 1])  # 90° rotation around z

Building from Basis Sets
------------------------

The :mod:`posym.tools` module provides utilities to construct basis
functions from PySCF or PyQchem basis set data:

.. code-block:: python

   from posym.tools import get_basis_set_pyscf, build_orbital

   # From a PySCF molecule object
   basis_set = get_basis_set_pyscf(mol_pyscf)

   # Build a molecular orbital
   orbital = build_orbital(basis_set, mo_coefficients)
