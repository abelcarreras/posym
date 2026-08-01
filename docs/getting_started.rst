Getting Started
===============

Installation
------------

PoSym can be installed directly from PyPI:

.. code-block:: bash

   pip install posym

Or from source:

.. code-block:: bash

   git clone https://github.com/abelcarreras/posym
   cd posym
   pip install -e .

.. note::

   PoSym includes C extensions for Gaussian integrals and permutation
   algorithms. These are compiled automatically during ``pip install``.

Dependencies
------------

- numpy
- scipy
- pandas
- PyYAML

Quick Example
-------------

Here is a minimal example creating symmetry objects and performing
direct products:

.. code-block:: python

   from posym import PointGroup, SymmetryObject

   pg = PointGroup(group='Td')
   print(pg)

   a1 = SymmetryObject(group='Td', rep='A1')
   t1 = SymmetryObject(group='Td', rep='T1')
   e = SymmetryObject(group='Td', rep='E')

   print('t1 * t1:', t1 * t1)
   print('t1 * e:', t1 * e)

For a more thorough introduction, see the :doc:`user_guide/index`.
