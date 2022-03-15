[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abelcarreras/posym/)

PoSym
=====
A python library for point symmetry operations

Requisites
----------
- numpy
- scipy
- pandas
- yaml

Basic example
-------------

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

Try a [full example](https://colab.research.google.com/github/abelcarreras/posym) in Google Colab


Note
----
This code is in a very early development and should be taken as a proof of concept