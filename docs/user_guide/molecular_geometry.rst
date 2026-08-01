Molecular Geometry
===================

The :class:`~posym.SymmetryMolecule` class computes the symmetry representation
of a molecular geometry with respect to a given point group.

Basic Usage
-----------

.. code-block:: python

   from posym import SymmetryMolecule

   coordinates = [[0.0, 0.0, 0.0],
                  [0.5541, 0.7996, 0.4965],
                  [0.6833, -0.8134, -0.2536],
                  [-0.7782, -0.3735, 0.6692],
                  [-0.4593, 0.3874, -0.9121]]

   symbols = ['C', 'H', 'H', 'H', 'H']

   sym = SymmetryMolecule(group='Td', coordinates=coordinates, symbols=symbols)

Continuous Symmetry Measure (CSM)
---------------------------------

The CSM quantifies the distance from a given geometry to the nearest
structure with perfect symmetry. It is returned by the :attr:`~posym.SymmetryMolecule.measure`
property:

.. math::

   S(G) = 100 \times \left(1 - \frac{1}{N_G} \sum_{g \in G} \frac{ \langle \mathbf{P} | g \mathbf{P} \rangle }{ \langle \mathbf{P} | \mathbf{P} \rangle } \right)

where :math:`\mathbf{P}` is the atomic position vector, :math:`G` is the
point group, and :math:`N_G` is the group order.

.. code-block:: python

   print('CSM (Td):', sym.measure)

Orientation
-----------

The molecule's orientation is optimized to maximize the symmetry overlap.
The optimal Euler angles (zyx convention) can be inspected:

.. code-block:: python

   angles = sym.orientation_angles  # [pitch, yaw, roll] in degrees

The optimization uses a two-step procedure:
 1. Pre-scan over a uniform distribution of angles (Fibonacci sphere)
 2. Refined conjugate-gradient minimization from the best guess

Symmetrized Coordinates
-----------------------

The :attr:`~posym.SymmetryMolecule.symmetrized_coordinates` property returns
coordinates averaged over all symmetry operations, i.e. the closest perfectly
symmetric structure:

.. code-block:: python

   coords = sym.symmetrized_coordinates

Configuration
-------------

The :class:`~posym.config.Configuration` class controls the optimization
parameters:

.. code-block:: python

   from posym.config import Configuration
   config = Configuration()
   config.algorithm = 'exact'        # brute-force permutation
   config.scan_steps = 20            # finer pre-scan
   config.fast_optimization = False  # full optimization

See :ref:`permutation-algorithms` for details on the algorithm options.
