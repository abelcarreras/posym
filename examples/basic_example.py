from posym import PointGroup, SymmetryBase

# Print symmetry IR table of C2h group
pg = PointGroup(group='C2h')
print('\nTest Point group table')
print(pg)

# define symmetry objects of particular IR in the C2h group
bu = SymmetryBase(group='C2h', rep='Bu')
bg = SymmetryBase('C2h', 'Bg')
ag = SymmetryBase('C2h', 'Ag')

# symmetry operation using symmetry objects
print('\nTest symmetry object operation')
print('bu * bg:', 2 * bu * bg)

print('(bu + bg) * (bu + bu + ag):', (bu + bg) * (bu + bu + ag))
state1 = (bu + bg) * (bu + bu + ag)

print('\nTest symmetry object representation')

# example with Td
pg = PointGroup(group='Td')
print(pg)

bg = SymmetryBase('C2h', 'Bg')
a1 = SymmetryBase(group='Td', rep='A1')
a2 = SymmetryBase(group='Td', rep='A2')
e = SymmetryBase(group='Td', rep='E')
t1 = SymmetryBase(group='Td', rep='T1')

print('e*e + a1:', e * e + 2 * a1)
print('t1*t1:', t1 * t1)

pg = PointGroup(group='C2v')
print('\nTest Point group table')
print(pg)

a1 = SymmetryBase(group='C2v', rep='A1')
b1 = SymmetryBase('C2v', 'B1')
a2 = SymmetryBase('C2v', 'A2')

print(a1 * a2 + b1)

# Example of convenient operators on symmetry objects
from posym.algebra import dot, norm

print('\nTest dot product')
print('A1 . A1: ', dot(a1, a1))
print('E . E : ', dot(e, e))
print('T1 . E : ', dot(t1, e))
print('T1 . (T1 + E): ', dot(t1, t1 + e))
print('T1 . (T1 + E) [normalized]: ', dot(t1, t1 + e, normalize=True))
print('(T1 + E) . (T1 + E): ', dot(t1 + e, t1 + e))
print('norm((T1 + E)): ', norm(t1 + e))
print('(T1 + E) . (T1 + E) [normalized]: ', dot(t1 + e, t1 + e, normalize=True))
print('(0.6T1 + E) . T1 : ', dot(0.6*t1 + e, t1))
print('(0.5T1 + E) . T1  [normalized]: ', dot(t1, t1, normalize=True))
