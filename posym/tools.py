from copy import deepcopy
import numpy as np
from posym.basis import BasisFunction, PrimitiveGaussian


def list_round(elements_list, decimals=2):
    r_list = []
    for element in elements_list:
        if abs(np.round(element) - element) < 10**(-decimals):
            r_list.append(int(np.round(element)))
        else:
            r_list.append(np.round(element, decimals))

    return r_list


def standardize_vector(vector, prec=1e-5):

    vector = np.array(vector, dtype=float)
    if np.abs(vector[0]) > prec:
        if vector[0] < 0:
            vector = np.array(vector) * -1
    elif np.abs(vector[1]) > prec:
        if vector[1] < 0:
            vector = np.array(vector) * -1
    else:
        if vector[2] < 0:
            vector = np.array(vector) * -1

    return vector.tolist()


def rotate_basis_set(basis_set, angle, axis):
    new_basis_set = deepcopy(basis_set)
    for bf in new_basis_set:
        bf.apply_rotation(angle, axis)
    return new_basis_set


def translate_basis_set(basis_set, translation):
    new_basis_set = deepcopy(basis_set)
    for bf in new_basis_set:
        bf.apply_translation(translation)
    return new_basis_set


def get_self_similarity(basis_set_1, density_matrix):
    from sympy.utilities.iterables import multiset_permutations
    n = len(basis_set_1)
    s = np.zeros((n, n, n, n))

    for i in range(n):
        for j in range(i+1):
            for k in range(j+1):
                for l in range(k+1):
                    integral = (basis_set_1[i] * basis_set_1[j] * basis_set_1[k] * basis_set_1[l]).integrate
                    for perm in multiset_permutations([i, j, k, l]):
                        dens_prod = density_matrix[perm[0], perm[1]] * density_matrix[perm[2], perm[3]]
                        s[perm[0], perm[1], perm[2], perm[3]] = integral * dens_prod

    return np.sum(s)


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


def get_basis_set(coordinates, basis, use_angstrom=True):
    """
    get list of basis functions (basis_set) from pyQchem formatted basis dictionary
    :param coordinates: list/array of atom positions
    :param basis: pyQchem basis dictionary
    :param use_angstrom: if True, coordinates are provided in angstrom else Bohr
    :return: list of BasisFunction objects
    """
    if use_angstrom:
        coordinates = np.array(coordinates)*1.8897259886  # convert from Angstrom to bohr

    basis_list = []
    for iatom, atom in enumerate(basis['atoms']):
        for shell in atom['shells']:
            if shell['shell_type'] == 's':
                primitives = []
                for exponent in shell['p_exponents']:
                    primitives.append(PrimitiveGaussian(alpha=exponent))

                basis_list.append(BasisFunction(primitives, shell['con_coefficients'],
                                                center=coordinates[iatom],
                                                label='{}:S'.format(atom['symbol'])))

            elif shell['shell_type'] == 'sp':
                primitives = []
                for exponent in shell['p_exponents']:
                    primitives.append(PrimitiveGaussian(alpha=exponent))

                basis_list.append(BasisFunction(primitives, shell['con_coefficients'],
                                                center=coordinates[iatom],
                                                label='{}:S'.format(atom['symbol'])))

                for l_set in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                    primitives = []
                    for exponent in shell['p_exponents']:
                        primitives.append(PrimitiveGaussian(alpha=exponent, l=l_set))

                    basis_list.append(BasisFunction(primitives, shell['p_con_coefficients'],
                                                    center=coordinates[iatom],
                                                    label='{}:P_{}'.format(atom['symbol'], l_set)))

            elif shell['shell_type'] == 'p':
                for l_set in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                    primitives = []
                    for exponent in shell['p_exponents']:
                        primitives.append(PrimitiveGaussian(alpha=exponent, l=l_set))

                    basis_list.append(BasisFunction(primitives, shell['con_coefficients'],
                                                    center=coordinates[iatom],
                                                    label='{}:P_{}'.format(atom['symbol'], l_set)))

            elif shell['shell_type'] == 'd':
                for l_set in [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]]:
                    primitives = []
                    for exponent in shell['p_exponents']:
                        primitives.append(PrimitiveGaussian(alpha=exponent, l=l_set))
                    basis_list.append(BasisFunction(primitives, shell['con_coefficients'],
                                                    center=coordinates[iatom],
                                                    label='{}:D_{}'.format(atom['symbol'], l_set)))

            elif shell['shell_type'] == 'd_':
                basis_list_temp = []
                for l_set in [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]]:
                    primitives = []
                    for exponent in shell['p_exponents']:
                        primitives.append(PrimitiveGaussian(alpha=exponent, l=l_set))
                    basis_list_temp.append(BasisFunction(primitives, shell['con_coefficients'],
                                                         center=coordinates[iatom],
                                                         label='{}:D_{}'.format(atom['symbol'], l_set)))

                # d0 : 2*z2-x2-y2
                d0 = 2 * basis_list_temp[2] - basis_list_temp[0] - basis_list_temp[1]
                norm = 1/np.sqrt((d0*d0).integrate)
                basis_list.append(d0 * norm)

                # d1 : xz
                d1 = basis_list_temp[4]
                basis_list.append(d1)

                # d2 : yz
                d2 = basis_list_temp[5]
                basis_list.append(d2)

                # d3 : x2-y2
                d3 = basis_list_temp[0] - basis_list_temp[1]
                norm = 1/np.sqrt((d3*d3).integrate)
                basis_list.append(d3 * norm)

                # d4 : xy
                d4 = basis_list_temp[3]
                basis_list.append(d4)

            elif shell['shell_type'] == 'f':
                for l_set in [[3, 0, 0], [2, 1, 0], [2, 0, 1], [1, 2, 0], [1, 1, 1],
                              [1, 0, 2], [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3]]:
                    primitives = []
                    for exponent in shell['p_exponents']:
                        primitives.append(PrimitiveGaussian(alpha=exponent, l=l_set))
                    basis_list.append(BasisFunction(primitives, shell['con_coefficients'],
                                                    center=coordinates[iatom],
                                                    label='{}:F_{}'.format(atom['symbol'], l_set)))

            elif shell['shell_type'] == 'f_':
                basis_list_temp = []
                              #  xxx 0     xxy 1      xxz 2      xyy 3      xyz 4
                for l_set in [[3, 0, 0], [2, 1, 0], [2, 0, 1], [1, 2, 0], [1, 1, 1],
                              #  xzz 5     yyy 6      yyz 7      yzz 8      zzz 9
                              [1, 0, 2], [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3]]:
                    primitives = []
                    for exponent in shell['p_exponents']:
                        primitives.append(PrimitiveGaussian(alpha=exponent, l=l_set))
                    basis_list_temp.append(BasisFunction(primitives, shell['con_coefficients'],
                                                         center=coordinates[iatom],
                                                         label='{}:F_{}'.format(atom['symbol'], l_set)))

                # f0 : zzz - 3.0/2.0 * (xxz + yyz)
                f0 = basis_list_temp[9] - 3/2*(basis_list_temp[2] + basis_list_temp[7])
                norm = 1/np.sqrt((f0*f0).integrate)
                basis_list.append(f0 * norm)

                # f1 : 6.0 * xzz - 3.0/2.0 * (xxx + xyy)
                f1 = 6 * basis_list_temp[5] - 3/2 * (basis_list_temp[0] + basis_list_temp[3])
                norm = 1/np.sqrt((f1*f1).integrate)
                basis_list.append(f1 * norm)

                # f2 : 6.0 * yzz - 3.0/2.0 * (xxy + yyy)
                f2 = 6 * basis_list_temp[8] - 3/2 * (basis_list_temp[1] + basis_list_temp[6])
                norm = 1/np.sqrt((f2*f2).integrate)
                basis_list.append(f2 * norm)

                # f3 : 15.0 * (xxz - yyz))
                f3 = 15 * (basis_list_temp[2] - basis_list_temp[7])
                norm = 1/np.sqrt((f3*f3).integrate)
                basis_list.append(f3 * norm)

                # f4 : 30.0 * xyz
                f4 = 30 * basis_list_temp[4]
                norm = 1/np.sqrt((f4*f4).integrate)
                basis_list.append(f4 * norm)

                # f5 : 15.0 * xxx - 45.0 * xyy
                f5 = 15 * basis_list_temp[0] - 45 * basis_list_temp[3]
                norm = 1/np.sqrt((f5*f5).integrate)
                basis_list.append(f5 * norm)

                # f6 : 45.0 * xxy - 15.0 * yyy
                f6 = 45 * basis_list_temp[1] - 15 * basis_list_temp[6]
                norm = 1/np.sqrt((f6*f6).integrate)
                basis_list.append(f6 * norm)

            else:
                raise Exception('Not implemented shell type:{}'.format(shell['shell_type']))

    return basis_list
