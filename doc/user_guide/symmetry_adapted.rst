Symmetry-Adapted Coordinates
=============================

The :class:`~posym.SymmetryAdaptedCoordinates` class generates
symmetry-adapted linear combinations (SALCs) of atomic displacements
from a molecular geometry.

.. note::

   This class currently only supports **Abelian groups** (groups where
   all irreducible representations are one-dimensional).

Basic Usage
-----------

.. code-block:: python

   from posym import SymmetryAdaptedCoordinates

   coordinates = [[0.0, 0.0, -0.0809],
                  [-1.4326, 0.0, -1.2824],
                  [1.4326, 0.0, -1.2824]]

   symbols = ['O', 'H', 'H']

   sac = SymmetryAdaptedCoordinates(group='c2v',
                                    coordinates=coordinates,
                                    symbols=symbols)

   for i in range(sac.get_number_of_modes()):
       print(f'SALC {i+1}: {sac.get_state_mode(i)}')

   coordinates = sac.get_symmetry_adapted_coordinates()

How It Works
------------

SALCs are generated using projection operators constructed from the group's
transformation matrix:

.. math::

   \hat{P}^{\Gamma} = \sum_{R \in G} \chi^{\Gamma}(R) \hat{R}

where :math:`\chi^{\Gamma}(R)` is the character of the irreducible
representation :math:`\Gamma` under operation :math:`R`. The projection
operators acting on the atomic displacement basis yield the symmetry-adapted
coordinates.

Only linearly independent projections are retained.
