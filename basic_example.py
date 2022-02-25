from posym import PointGroup, SymmetryBase

pg = PointGroup(group='C2h')
print('\nTest Point group table')
print(pg)

bu = SymmetryBase(group='C2h', rep='Bu')
bg = SymmetryBase('C2h', 'Bg')
ag = SymmetryBase('C2h', 'Ag')

print('\nTest symmetry object operation')
print('bu * bg:', 2 * bu * bg)

print('(bu + bg) * (bu + bu + ag):', (bu + bg) * (bu + bu + ag))
state1 = (bu + bg) * (bu + bu + ag)

print('\nTest symmetry object representation')

pg = PointGroup(group='Td')
print(pg)

bg = SymmetryBase('C2h', 'Bg')
a1 = SymmetryBase(group='Td', rep='A1')
a2 = SymmetryBase(group='Td', rep='A2')
e = SymmetryBase(group='Td', rep='E')
t1 = SymmetryBase(group='Td', rep='T1')

print('e*e + a1:', e * e + 2 * a1)


pg = PointGroup(group='C2v')
print('\nTest Point group table')
print(pg)

a1 = SymmetryBase(group='C2v', rep='A1')
b1 = SymmetryBase('C2v', 'B1')
a2 = SymmetryBase('C2v', 'A2')

print(a1 * a2 + b1)

from posym.algebra import dot

print('\nTest dot product')
print('A1 * A1: ', dot(a1, a1))
print('E * E : ', dot(e, e))
print('T1 * E : ', dot(t1, e))
print('T1 * (T1 + E): ', dot(t1, t1 + e))
print('T1 * (T1 + E) [normalized]: ', dot(t1, t1 + e, normalize=True))
print('(T1 + E) * (T1 + E): ', dot(t1 + e, t1 + e))
print('(T1 + E) * (T1 + E) [normalized]: ', dot(t1 + e, t1 + e, normalize=True))
print('(0.6T1 + E) * T1  [projection]: ', dot(0.6*t1 + e, t1, projection=True))
print('(0.5T1 + E) * T1  [projection, normalized]: ', dot(0.5*t1 + e, t1, projection=True, normalize=True))
