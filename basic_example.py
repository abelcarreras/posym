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
