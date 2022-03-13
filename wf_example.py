from posym.basis import PrimitiveGaussian, BasisFunction
from posym import SymmetryFunction
import numpy as np
import posym.algebra as al


basis = {'name': 'STO-3G',
         'primitive_type': 'gaussian',
         'atoms': [{'symbol': 'O',
                    'shells': [{'shell_type': 's',
                                'p_exponents': [130.70932, 23.808861, 6.4436083],
                                'con_coefficients': [0.154328969, 0.535328136, 0.444634536],
                                'p_con_coefficients': [0.0, 0.0, 0.0]},
                               {'shell_type': 'sp',
                                'p_exponents': [5.0331513, 1.1695961, 0.380389],
                                'con_coefficients': [-0.0999672287, 0.399512825, 0.700115461],
                                'p_con_coefficients': [0.155916268, 0.607683714, 0.391957386]}]},
                   {'symbol': 'H',
                    'shells': [{'shell_type': 's',
                                'p_exponents': [3.42525091, 0.62391373, 0.1688554],
                                'con_coefficients': [0.154328971, 0.535328142, 0.444634542],
                                'p_con_coefficients': [0.0, 0.0, 0.0]}]},
                   {'symbol': 'H',
                    'shells': [{'shell_type': 's',
                                'p_exponents': [3.42525091, 0.62391373, 0.1688554],
                                'con_coefficients': [0.154328971, 0.535328142, 0.444634542],
                                'p_con_coefficients': [0.0, 0.0, 0.0]}]}]}

mo_coefficients = [[ 0.994216442,  0.025846814, 0.000000000,  0.000000000, -0.004164076, -0.005583712, -0.005583712],
                   [ 0.233766661, -0.844456594, 0.000000000,  0.000000000,  0.122829781, -0.155593214, -0.155593214],
                   [ 0.000000000,  0.000000000, 0.612692349,  0.000000000,  0.000000000, -0.449221684,  0.449221684],
                   [-0.104033343,  0.538153649, 0.000000000,  0.000000000,  0.755880259, -0.295107107, -0.295107107],
                   [ 0.000000000,  0.000000000, 0.000000000, -1.000000000,  0.000000000,  0.000000000,  0.000000000],
                   [-0.125818566,  0.820120983, 0.000000000,  0.000000000, -0.763538862, -0.769155124, -0.769155124],
                   [ 0.000000000,  0.000000000, 0.959800163,  0.000000000,  0.000000000,  0.814629717, -0.814629717]]

# coordinates in bohr
coordinates = [[ 0.00000, 0.0000000, -0.0808819],
               [-1.43262, 0.0000000, -1.2823700],
               [ 1.43262, 0.0000000, -1.2823700]]

symbols = ['O', 'H', 'H']

# Oxigen atom
sa = PrimitiveGaussian(alpha=130.70932)
sb = PrimitiveGaussian(alpha=23.808861)
sc = PrimitiveGaussian(alpha=6.4436083)
s_O = BasisFunction([sa, sb, sc],
                    [0.154328969, 0.535328136, 0.444634536],
                    coordinates=[0.0000000000, 0.000000000, -0.0808819])

sa = PrimitiveGaussian(alpha=5.03315132)
sb = PrimitiveGaussian(alpha=1.1695961)
sc = PrimitiveGaussian(alpha=0.3803890)
s2_O = BasisFunction([sa, sb, sc],
                     [-0.099967228, 0.399512825, 0.700115461],
                     coordinates=[0.0000000000, 0.000000000, -0.0808819])

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
                     coordinates=[0.0000000000, 0.000000000, -0.0808819])
py_O = BasisFunction([pya, pyb, pyc],
                     [0.155916268, 0.6076837186, 0.3919573931],
                     coordinates=[0.0000000000, 0.000000000, -0.0808819])
pz_O = BasisFunction([pza, pzb, pzc],
                     [0.155916268, 0.6076837186, 0.3919573931],
                     coordinates=[0.0000000000, 0.000000000, -0.0808819])

# Hydrogen atoms
sa = PrimitiveGaussian(alpha=3.42525091)
sb = PrimitiveGaussian(alpha=0.62391373)
sc = PrimitiveGaussian(alpha=0.1688554)
s_H = BasisFunction([sa, sb, sc],
                    [0.154328971, 0.535328142, 0.444634542],
                    coordinates=[-1.43262, 0.000000000, -1.28237])

s2_H = BasisFunction([sa, sb, sc],
                     [0.154328971, 0.535328142, 0.444634542],
                     coordinates=[1.43262, 0.000000000, -1.28237])

basis_functions = [s_O, s2_O, px_O, py_O, pz_O, s_H, s2_H]


density_matrix = 0 * np.outer(mo_coefficients[0], mo_coefficients[0]) + \
                 0 * np.outer(mo_coefficients[1], mo_coefficients[1]) + \
                 2 * np.outer(mo_coefficients[2], mo_coefficients[2]) + \
                 2 * np.outer(mo_coefficients[3], mo_coefficients[3]) + \
                 0 * np.outer(mo_coefficients[4], mo_coefficients[4]) + \
                 0 * np.outer(mo_coefficients[5], mo_coefficients[5]) + \
                 0 * np.outer(mo_coefficients[6], mo_coefficients[6])

def build_density(basis_set, density_matrix):
    density = BasisFunction([], [])
    for i, basis1 in enumerate(basis_set):
        for j, basis2 in enumerate(basis_set):
            density += basis1*basis2 * density_matrix[i, j]

    return density

def build_density_short(basis_set, density_matrix):
    density = BasisFunction([], [])
    n_functions = len(basis_set)
    for i in range(n_functions):
        for j in range(i+1, n_functions):
            density += 2.0*basis_set[i]*basis_set[j] * density_matrix[i, j]

    for i, basis in enumerate(basis_set):
        density += basis * basis * density_matrix[i, i]

    return density

def build_orbital(basis_set, mo_coefficients):
    orbital = BasisFunction([], [])
    for mo_coeff, basis in zip(mo_coefficients, basis_set):
        orbital += mo_coeff * basis

    return orbital


orbital_1 = build_orbital(basis_functions, mo_coefficients[0])
orbital_2 = build_orbital(basis_functions, mo_coefficients[1])
orbital_3 = build_orbital(basis_functions, mo_coefficients[2])
orbital_4 = build_orbital(basis_functions, mo_coefficients[3])
orbital_5 = build_orbital(basis_functions, mo_coefficients[4])

sym_o1 = SymmetryFunction('c2v', coordinates, symbols, orbital_1)
sym_o2 = SymmetryFunction('c2v', coordinates, symbols, orbital_2)
sym_o3 = SymmetryFunction('c2v', coordinates, symbols, orbital_3)
sym_o4 = SymmetryFunction('c2v', coordinates, symbols, orbital_4)
sym_o5 = SymmetryFunction('c2v', coordinates, symbols, orbital_5)

print('Symmetry O1: ', sym_o1)
print('Symmetry O2: ', sym_o2)
print('Symmetry O3: ', sym_o3)
print('Symmetry O4: ', sym_o4)
print('Symmetry O5: ', sym_o5)

f_density = build_density_short(basis_functions, density_matrix)
print('density integral: ', f_density.integrate)

sym_density = SymmetryFunction('c2v', coordinates, symbols, f_density)
print('Symmetry density: ', sym_density)
print('Symmetry norm: ', al.norm(sym_density))
print('density self_similarity', sym_density.self_similarity)