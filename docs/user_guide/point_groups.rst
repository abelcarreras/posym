Point Groups
============

The :class:`~posym.PointGroup` class provides access to point group
character tables and symmetry operations.

Creating a Point Group
----------------------

.. code-block:: python

   from posym import PointGroup

   pg = PointGroup(group='Td')
   print(pg)

Labels are **case-insensitive**: ``'Td'``, ``'td'``, and ``'TD'`` are all
equivalent.

Supported Groups
----------------

The following point groups are available:

- **Special groups:** ``Cs``, ``Ci``, ``T``, ``Td``, ``Th``, ``O``, ``Oh``,
  ``I``, ``Ih``
- **Generic groups (any n):** ``C{n}``, ``C{n}v``, ``C{n}h``, ``D{n}``,
  ``D{n}h``, ``D{n}d``, ``S{n}``

For generic groups, tables are generated programmatically for any integer
``{n}``.

Group Properties
----------------

.. code-block:: python

   pg = PointGroup('Oh')

   pg.order           # 48 — group order
   pg.n_ir            # 10 — number of irreducible representations
   pg.is_abelian      # False
   pg.ir_labels       # ['A1g', 'A2g', 'Eg', 'T1g', 'T2g', ...]
   pg.op_labels       # ['E', 'C3', 'C2', 'C4', 'C2prime', 'i', ...]
   pg.ir_degeneracies # [1, 1, 2, 3, 3, 1, 1, 2, 3, 3]

Operations
----------

The group contains symmetry operations represented as 3x3 transformation
matrices:

.. code-block:: python

   # Class representatives
   for op in pg.operations:
       print(op.label, op.order, op.matrix_representation)

   # All operations (including degenerate copies)
   all_ops = pg.all_operations

   # Generator operations
   for gen in pg.generators:
       print(gen.label, gen.matrix_representation)


Transform Matrices
------------------

PoSym uses the transformation matrix M defines for each symmetry group to convert between the operator basis
and the irreducible representation basis (See :ref:`transform-matrices` for further details).
These are accessible via:

.. code-block:: python

   M = pg.trans_matrix           # IR → Op
   M_inv = pg.trans_matrix_inv   # Op → IR
