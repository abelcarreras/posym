Normal Modes
============

The :class:`~posym.SymmetryNormalModes` class determines the symmetry representation of vibrational normal modes.

Basic Usage
-----------

.. code-block:: python

   from posym import SymmetryNormalModes

   coordinates = [[0.0, 0.0, -0.0809],
                  [-1.4326, 0.0, -1.2824],
                  [1.4326, 0.0, -1.2824]]

   symbols = ['O', 'H', 'H']

   modes = [[[0., 0., -0.075],
             [-0.381, -0., 0.593],
             [0.381, -0., 0.593]],

            [[-0., -0., 0.044],
             [-0.613, -0., -0.35],
             [0.613, 0., -0.35]],

            [[-0.073, -0., -0.],
             [0.583, 0., 0.397],
             [0.583, 0., -0.397]]]

   sm = SymmetryNormalModes(group='c2v', coordinates=coordinates,
                            modes=modes, symbols=symbols)

   for i in range(sm.get_number_of_modes()):
       print(f'Mode {i+1}: {sm.get_state_mode(i)}')

   print('Total:', sm)


:meth:`~posym.SymmetryNormalModes.get_state_mode` returns the symmetry representation of
each mode as a :class:`~posym.SymmetryObject`.

Atom Displacements
------------------

:class:`~posym.SymmetryAtomDisplacements` computes :math:`\Gamma_{3N}`,
the total representation of all atomic displacements (sum of all normal
mode symmetries):

.. code-block:: python

   from posym import SymmetryAtomDisplacements

   gamma_3n = SymmetryAtomDisplacements(group='c2v',
                                        coordinates=coordinates,
                                        symbols=symbols)

   print('Gamma_3N:', gamma_3n)

This representation decomposes as:

.. math::

   \Gamma_{3N} = \Gamma_{\text{trans}} + \Gamma_{\text{rot}} + \Gamma_{\text{vib}}

where :math:`\Gamma_{\text{vib}}` is the total normal mode representation.
