Symmetry Objects
================

The :class:`~posym.SymmetryObject` is the core abstraction in PoSym. It
represents an element of the G-module space, a vector in the basis of
irreducible representations of a point group.

Creating Symmetry Objects
-------------------------

From an IR label:

.. code-block:: python

   from posym import SymmetryObject

   a1 = SymmetryObject(group='Td', rep='A1')
   e = SymmetryObject(group='Td', rep='E')
   t2 = SymmetryObject(group='Td', rep='T2')

Alternative: Using the ``from_label`` class method:

.. code-block:: python

   b1 = SymmetryObject.from_label(group='C2v', rep='B1')

Direct Sum and Direct Product
-----------------------------

Symmetry objects can be combined using ``+`` (direct sum) and ``*`` (direct
product):

.. code-block:: python

   # Direct sum: A1 + E
   combined = a1 + e

   # Direct product: T2 x T2 = A1 + E + T1 + T2
   product = t2 * t2

   # Scalar multiplication
   scaled = 2 * e

The resulting object decomposes into its irreducible representation
components automatically.

Inspecting Representations
--------------------------

.. code-block:: python

   state = t1 * t1  # T1 x T1 in Td

   # Get the irreducible representation vector
   ir_rep = state.get_ir_representation()
   # A1: 1.0, A2: 0.0, E: 1.0, T1: 1.0, T2: 1.0

   # Get the raw operator-character vector
   op_rep = state.get_op_representation()

   # Print as human-readable string
   print(state)  # A1 + E + T1 + T2

Algebra Utilities
-----------------

The :mod:`posym.algebra` module provides inner product and norm operations
in the space of symmetry representations:

.. code-block:: python

   from posym import algebra as al

   al.dot(t1, t1)           # 9.0
   al.dot(t1, t1, normalize=True)  # 1.0
   al.norm(t1 + e)          # 5.0

The :func:`~posym.algebra.dot` function computes the squared inner product
weighted by irreducible representation degeneracies, corresponding to the
overlap in the G-module space.
