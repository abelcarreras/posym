[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abelcarreras/posym/)

PoSym
=====

A simple point symmetry analysis tool written in python.  
This tool is mainly designed for theoretical chemistry but 
can be used in general for other objects by defining subclasses 
of the `SymmetryBase` class. At this point `SymmetryModes` and 
`SymmetryFunction` subclasses are defined for the analysis of 
vibrational normal modes and molecular orbitals respectively.

Features
--------
- Use as simple calculator for irreducible representations supporting direct sum and product
- Handles pseudosymmetry
- Determine point symmetry from normal modes 
- Determine point symmetry functions defined in gaussian basis
- Compatibility with PyQchem (http://www.github.com/abelcarreras/pyqchem)

Requisites
----------
- numpy
- scipy
- pandas
- yaml

Use as a simple symmetry calculation
------------------------------------
posym allows to create symmetry python objects that can be operated using usual operators
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

Determine the symmetry of normal modes
--------------------------------------
Symmetry objects can be obtained from normal modes
```python
from posym import SymmetryModes

coordinates = [[ 0.00000, 0.0000000, -0.0808819],
               [-1.43262, 0.0000000, -1.2823700],
               [ 1.43262, 0.0000000, -1.2823700]]

symbols = ['O', 'H', 'H']

normal_modes = [[[ 0.,     0.,    -0.075],
                 [-0.381, -0.,     0.593],
                 [ 0.381, -0.,     0.593]], # mode 1

                [[-0.   , -0.,     0.044],
                 [-0.613, -0.,    -0.35 ],
                 [ 0.613,  0.,    -0.35 ]], # mode 2

                [[-0.073, -0.,    -0.   ],
                 [ 0.583,  0.,     0.397],
                 [ 0.583,  0.,    -0.397]]] # mode 3

frequencies = [1737.01, 3988.5, 4145.43]

sym_modes_gs = SymmetryModes(group='c2v', coordinates=coordinates, modes=normal_modes, symbols=symbols)
for i in range(len(normal_modes)):
    print('Mode {:2}: {:8.3f} :'.format(i + 1, frequencies[i]), sym_modes_gs.get_state_mode(i))

print('Total symmetry: ', sym_modes_gs)

```

Define basis set functions in gaussian basis
--------------------------------------------
define basis function as linear combination of gaussian that act as normal python functions 
```python
from posym.basis import PrimitiveGaussian, BasisFunction

# Oxigen atom
sa = PrimitiveGaussian(alpha=130.70932)
sb = PrimitiveGaussian(alpha=23.808861)
sc = PrimitiveGaussian(alpha=6.4436083)
s_O = BasisFunction([sa, sb, sc],
                    [0.154328969, 0.535328136, 0.444634536],
                    center=[0.0000000000, 0.000000000, -0.0808819]) # Bohr

sa = PrimitiveGaussian(alpha=5.03315132)
sb = PrimitiveGaussian(alpha=1.1695961)
sc = PrimitiveGaussian(alpha=0.3803890)
s2_O = BasisFunction([sa, sb, sc],
                     [-0.099967228, 0.399512825, 0.700115461],
                     center=[0.0000000000, 0.000000000, -0.0808819])

pxa = PrimitiveGaussian(alpha=5.0331513, l=[1, 0, 0])
pxb = PrimitiveGaussian(alpha=1.1695961, l=[1, 0, 0])
pxc = PrimitiveGaussian(alpha=0.3803890, l=[1, 0, 0])

pya = PrimitiveGaussian(alpha=5.0331513, l=[0, 1, 0])
pyb = PrimitiveGaussian(alpha=1.1695961, l=[0, 1, 0])
pyc = PrimitiveGaussian(alpha=0.3803890, l=[0, 1, 0])

pza = PrimitiveGaussian(alpha=5.0331513, l=[0, 0, 1])
pzb = PrimitiveGaussian(alpha=1.1695961, l=[0, 0, 1])
pzc = PrimitiveGaussian(alpha=0.3803890, l=[0, 0, 1])

px_O = BasisFunction([pxa, pxb, pxc],
                     [0.155916268, 0.6076837186, 0.3919573931],
                     center=[0.0000000000, 0.000000000, -0.0808819])
py_O = BasisFunction([pya, pyb, pyc],
                     [0.155916268, 0.6076837186, 0.3919573931],
                     center=[0.0000000000, 0.000000000, -0.0808819])
pz_O = BasisFunction([pza, pzb, pzc],
                     [0.155916268, 0.6076837186, 0.3919573931],
                     center=[0.0000000000, 0.000000000, -0.0808819])

# Hydrogen atoms
sa = PrimitiveGaussian(alpha=3.42525091)
sb = PrimitiveGaussian(alpha=0.62391373)
sc = PrimitiveGaussian(alpha=0.1688554)
s_H = BasisFunction([sa, sb, sc],
                    [0.154328971, 0.535328142, 0.444634542],
                    center=[-1.43262, 0.000000000, -1.28237])

s2_H = BasisFunction([sa, sb, sc],
                     [0.154328971, 0.535328142, 0.444634542],
                     center=[1.43262, 0.000000000, -1.28237])

basis_set = [s_O, s2_O, px_O, py_O, pz_O, s_H, s2_H]

# Operate with basis functions in analytic form

px_O2 = px_O * px_O
print('integral from -inf to inf:', px_O2.integrate)

# plot functions
from matplotlib import pyplot as plt
import numpy as np

xrange = np.linspace(-5, 5, 100)
plt.plot(xrange, [s_O(x, 0, 0) for x in xrange] , label='s_O')
plt.plot(xrange, [px_O(x, 0, 0) for x in xrange] , label='px_O')
plt.legend()

```

Create molecular orbitals from basis set
----------------------------------------
Define molecular orbitals straightforwardly from molecular orbitals coefficients using usual operators
```python

# Orbital 1
o1 = s_O * 0.994216442 + s2_O * 0.025846814 + px_O * 0.0 + py_O * 0.0 + pz_O * -0.004164076 + s_H * -0.005583712 + s2_H * -0.005583712

# Orbital 2
o2 = s_O * 0.23376666 + s2_O * -0.844456594 + px_O * 0.0 + py_O * 0.0 + pz_O * 0.122829781 + s_H * -0.155593214 + s2_H * -0.155593214

# Orbital 3
o3 = s_O * 0.0 + s2_O * 0.0 + px_O * 0.612692349 + py_O * 0.0 + pz_O * 0.0 + s_H * -0.44922168 + s2_H * 0.449221684

# Orbital 4
o4 = s_O * -0.104033343 + s2_O * 0.538153649 + px_O * 0.0 + py_O * 0.0 + pz_O * 0.755880259 + s_H * -0.295107107 + s2_H * -0.2951071074

# Orbital 5
o5 = s_O * 0.0 + s2_O * 0.0 + px_O * 0.0 + py_O * -1.0 + pz_O * 0.0 + s_H * 0.0 + s2_H * 0.0

# Orbital 6
o6 = s_O * -0.125818566 + s2_O * 0.820120983 + px_O * 0.0 + py_O * 0.0 + pz_O * -0.763538862 + s_H * -0.769155124 + s2_H * -0.769155124


# Check orthogonality
print('<o1|o1>: ', (o1*o1).integrate)
print('<o2|o2>: ', (o2*o2).integrate)
print('<o1|o2>: ', (o1*o2).integrate)
```

Analyze symmetry of molecular orbitals
--------------------------------------
Get symmetry objects from PrimitiveGaussian/BasisFunction type objects
```python
from posym import SymmetryFunction

sym_o1 = SymmetryFunction('c2v', o1)
sym_o2 = SymmetryFunction('c2v', o2)
sym_o3 = SymmetryFunction('c2v', o3)
sym_o4 = SymmetryFunction('c2v', o4)
sym_o5 = SymmetryFunction('c2v', o5)
sym_o6 = SymmetryFunction('c2v', o6)

print('Symmetry O1: ', sym_o1)
print('Symmetry O2: ', sym_o2)
print('Symmetry O3: ', sym_o3)
print('Symmetry O4: ', sym_o4)
print('Symmetry O5: ', sym_o5)
print('Symmetry O6: ', sym_o6)

# Operate molecular orbitals symmetries to get the symmetry of wave functions

# restricted close shell
sym_wf_gs = sym_o1*sym_o1 * sym_o2*sym_o2 * sym_o3*sym_o3 * sym_o4*sym_o4 * sym_o5*sym_o5
print('Symmetry WF (ground state): ', sym_wf_gs)

# restricted open shell
sym_wf_excited_1 = sym_o1*sym_o1 * sym_o2*sym_o2 * sym_o3*sym_o3 * sym_o4*sym_o4 * sym_o5*sym_o6
print('Symmetry WF (excited state 1): ', sym_wf_excited_1)

# restricted close shell
sym_wf_excited_2 = sym_o1*sym_o1 * sym_o2*sym_o2 * sym_o3*sym_o3 * sym_o4*sym_o4 * sym_o6*sym_o6
print('Symmetry WF (excited state 2): ', sym_wf_excited_2)

```

Combine with PyQchem to create automatic workflows
--------------------------------------------------
```python

from pyqchem import get_output_from_qchem, QchemInput, Structure
from pyqchem.parsers.basic import basic_parser_qchem
from posym import SymmetryFunction
# convenient functions to connect pyqchem - posym
from posym.tools import get_basis_set, build_orbital 

# define molecules
butadiene_cis = Structure(coordinates=[[ -1.07076839,   -2.13175980,    0.03234382],
                                       [ -0.53741536,   -3.05918866,    0.04995793],
                                       [ -2.14073783,   -2.12969357,    0.04016267],
                                       [ -0.39112115,   -0.95974916,    0.00012984],
                                       [  0.67884827,   -0.96181542,   -0.00769025],
                                       [ -1.15875076,    0.37505495,   -0.02522296],
                                       [ -0.62213437,    1.30041753,   -0.05065831],
                                       [ -2.51391203,    0.37767199,   -0.01531698],
                                       [ -3.04726506,    1.30510083,   -0.03293196],
                                       [ -3.05052841,   -0.54769055,    0.01011971]],
                          symbols=['C', 'H', 'H', 'C', 'H', 'C', 'H', 'C', 'H', 'H'])


# create qchem input
qc_input = QchemInput(butadiene_cis,
                      jobtype='sp',
                      exchange='hf',
                      basis='sto-3g',
                      )

# calculate and parse qchem output
data_cis, ee_cis = get_output_from_qchem(qc_input,
                                         read_fchk=True,
                                         processors=4,
                                         parser=basic_parser_qchem)

# extract required information from Q-Chem calculation
coordinates = ee_cis['structure'].get_coordinates()
mo_coefficients = ee_cis['coefficients']
basis = ee_cis['basis']

# print results
print('Molecular orbitals (alpha) symmetry')
basis_set = get_basis_set(coordinates, basis)
for i, orbital_coeff in enumerate(mo_coefficients['alpha']):
    orbital = build_orbital(basis_set, orbital_coeff)
    sym_orbital = SymmetryFunction('c2v', orbital)
    print('Symmetry O{}: '.format(i+1), sym_orbital)
    

```


Try an [interactive example](https://colab.research.google.com/github/abelcarreras/posym) in Google Colab

