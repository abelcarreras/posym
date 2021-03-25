PoSym
=====
A python library for point symmetry operations

Requisites
----------
numpy
yaml

Example
-------

```python
from posym import PointGroup, SymmetryBase

pg = PointGroup(group='Td')
print(pg)

a1 = SymmetryBase(group='Td', rep='A1')
a2 = SymmetryBase(group='Td', rep='A2')
e = SymmetryBase(group='Td', rep='E')
t1 = SymmetryBase(group='Td', rep='T1')

print('e*e + a1:', e * (e + a1))

```
