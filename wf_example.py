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
                    center=coordinates[0])

sa = PrimitiveGaussian(alpha=5.03315132)
sb = PrimitiveGaussian(alpha=1.1695961)
sc = PrimitiveGaussian(alpha=0.3803890)
s2_O = BasisFunction([sa, sb, sc],
                     [-0.099967228, 0.399512825, 0.700115461],
                     center=coordinates[0])

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
                     center=coordinates[0])
py_O = BasisFunction([pya, pyb, pyc],
                     [0.155916268, 0.6076837186, 0.3919573931],
                     center=coordinates[0])
pz_O = BasisFunction([pza, pzb, pzc],
                     [0.155916268, 0.6076837186, 0.3919573931],
                     center=[0.0000000000, 0.000000000, -0.0808819])

# Hydrogen atoms
sa = PrimitiveGaussian(alpha=3.42525091)
sb = PrimitiveGaussian(alpha=0.62391373)
sc = PrimitiveGaussian(alpha=0.1688554)
s_H = BasisFunction([sa, sb, sc],
                    [0.154328971, 0.535328142, 0.444634542],
                    center=coordinates[1])

s2_H = BasisFunction([sa, sb, sc],
                     [0.154328971, 0.535328142, 0.444634542],
                     center=coordinates[2])

basis_functions = [s_O, s2_O, px_O, py_O, pz_O, s_H, s2_H]


density_matrix = 0 * np.outer(mo_coefficients[0], mo_coefficients[0]) + \
                 0 * np.outer(mo_coefficients[1], mo_coefficients[1]) + \
                 2 * np.outer(mo_coefficients[2], mo_coefficients[2]) + \
                 2 * np.outer(mo_coefficients[3], mo_coefficients[3]) + \
                 0 * np.outer(mo_coefficients[4], mo_coefficients[4]) + \
                 0 * np.outer(mo_coefficients[5], mo_coefficients[5]) + \
                 0 * np.outer(mo_coefficients[6], mo_coefficients[6])

def build_density(basis_set, density_matrix):
    density_matrix = np.array(density_matrix)
    density = BasisFunction([], [])
    for i, basis1 in enumerate(basis_set):
        for j, basis2 in enumerate(basis_set):
            density += basis1*basis2 * density_matrix[i, j]

    return density

def build_orbital(basis_set, mo_coefficients):
    orbital = BasisFunction([], [])
    for mo_coeff, basis in zip(mo_coefficients, basis_set):
        orbital += mo_coeff * basis

    return orbital

dipole_x = [[          0.,          0.,  5.0792e-02,   0.,          0., -2.8130e-03,   2.8130e-03],
            [          0.,          0.,  6.4117e-01,   0.,          0., -2.7580e-01,   2.7580e-01],
            [  5.0792e-02,  6.4117e-01,          0.,   0.,          0.,  4.7462e-01,   4.7462e-01],
            [          0.,          0.,          0.,   0.,          0.,          0.,           0.],
            [          0.,          0.,          0.,   0.,          0.,  1.5329e-01,  -1.5329e-01],
            [ -2.8130e-03, -2.7580e-01,  4.7462e-01,   0.,  1.5329e-01, -1.4326e+00,   6.9389e-18],
            [  2.8130e-03,  2.7580e-01,  4.7462e-01,   0., -1.5329e-01,  6.9389e-18,   1.4326e+00]]


dipole_y = [[     0.,     0.,     0.,   0.0508,     0.,     0.,     0.],
            [     0.,     0.,     0.,   0.6412,     0.,     0.,     0.],
            [     0.,     0.,     0.,       0.,     0.,     0.,     0.],
            [ 0.0508, 0.6412,     0.,       0.,     0., 0.2918, 0.2918],
            [     0.,     0.,     0.,       0.,     0.,     0.,     0.],
            [     0.,     0.,     0.,   0.2918,     0.,     0.,     0.],
            [     0.,     0.,     0.,   0.2918,     0.,     0.,     0.]]

dipole_z = [[  -0.0809,  -0.0191,      0.,      0.,   0.0508,  -0.0064,  -0.0064],
            [  -0.0191,  -0.0809,      0.,      0.,   0.6412,  -0.2680,  -0.2680],
            [       0.,       0., -0.0809,      0.,       0.,   0.1770,  -0.1770],
            [       0.,       0.,      0., -0.0809,       0.,       0.,       0.],
            [   0.0508,   0.6412,      0.,      0.,  -0.0809,   0.4403,   0.4403],
            [  -0.0064,  -0.2680,  0.1770,      0.,   0.4403,  -1.2824,  -0.3217],
            [  -0.0064,  -0.2680, -0.1770,      0.,   0.4403,  -0.3217,  -1.2824]]


fock_matrix = [[-20.242907978346196, -5.163857371732694, -3.075065754527229e-16, 1.8323635972251456e-16, 0.028693754702171125, -1.108332187645028, -1.1083321876450283],
               [-5.163857371732694, -2.439690414788265, -1.6261798133412796e-16, -1.496562668754178e-17, 0.11807009928392057, -0.971758748616808, -0.9717587486168073],
               [-3.075065754527229e-16, -1.6261798133412796e-16, -0.2938600994977424, 1.0117677673805833e-16, -1.9479752048417142e-16, 0.3784077931020735, -0.3784077931020744],
               [ 1.8323635972251456e-16, -1.496562668754178e-17, 1.0117677673805833e-16, -0.39261462735921066, 1.703969857431259e-18, -2.0264868649535489e-16, 5.016795557159616e-18],
               [ 0.028693754702171125, 0.11807009928392057, -1.9479752048417142e-16, 1.703969857431259e-18, -0.33633212327437595, 0.3734802991314477, 0.3734802991314475],
               [-1.108332187645028, -0.971758748616808, 0.3784077931020735, -2.0264868649535489e-16, 0.3734802991314477, -0.5411484905625149, -0.375307075852357],
               [-1.1083321876450283, -0.9717587486168073, -0.3784077931020744, 5.016795557159616e-18, 0.3734802991314475, -0.375307075852357, -0.541148490562514]]


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

f_density = build_density(basis_functions, density_matrix)
print('density integral: ', f_density.integrate)

sym_density = SymmetryFunction('c2v', coordinates, symbols, f_density)
print('Symmetry density: ', sym_density)
print('density self_similarity', sym_density.self_similarity)

f_dipole_x = build_density(basis_functions, dipole_x)
f_dipole_y = build_density(basis_functions, dipole_y)
f_dipole_z = build_density(basis_functions, dipole_z)

sym_dipole_x = SymmetryFunction('c2v', coordinates, symbols, f_dipole_x)
sym_dipole_y = SymmetryFunction('c2v', coordinates, symbols, f_dipole_y)
sym_dipole_z = SymmetryFunction('c2v', coordinates, symbols, f_dipole_z)

print('Symmetry dipole X operator: ', sym_dipole_x)
print('Symmetry dipole Y operator: ', sym_dipole_y)
print('Symmetry dipole Z operator: ', sym_dipole_z)

f_fock = build_density(basis_functions, fock_matrix)
sym_fock = SymmetryFunction('c2v', coordinates, symbols, f_fock)
print('Symmetry Fock operator: ', sym_fock)
