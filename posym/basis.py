import numpy as np
from copy import deepcopy
import math
import itertools
from posym.integrals import product_poly_coeff, gaussian_integral


def binomial_expansion(l, x, max_lim=None):

    if max_lim is None:
        max_lim = np.max(l)+1

    vector_x = np.zeros((max_lim, max_lim, max_lim))
    vector_y = np.zeros((max_lim, max_lim, max_lim))
    vector_z = np.zeros((max_lim, max_lim, max_lim))

    for k in range(l[0]+1):
        vector_x[k, 0, 0] += math.comb(l[0], k) * x[0]**(l[0]-k)
    for k in range(l[1] + 1):
        vector_y[0, k, 0] += math.comb(l[1], k) * x[1]**(l[1]-k)
    for k in range(l[2] + 1):
        vector_z[0, 0, k] += math.comb(l[2], k) * x[2]**(l[2]-k)

    vector = product_poly_coeff(product_poly_coeff(vector_x, vector_y), vector_z, max_lim)
    return vector


def product_poly_coeff_py(poly_coeff, poly_coeff2, max_lim=None):
    """
    Product of two polynomial coefficients matrices

    :param poly_coeff: matrix of polynomial coefficients 1
    :param poly_coeff2: matrix of polynomial coefficients 2
    :param max_lim:
    :return:
    """
    # max_lim = 0 if max_lim is None else max_lim
    # return product_poly_coeff_c(poly_coeff, poly_coeff2, max_lim)

    max_lim_1 = len(poly_coeff)
    max_lim_2 = len(poly_coeff2)
    max_lim_prod = max_lim_1 + max_lim_2
    poly_coeff_prod = np.zeros((max_lim_prod, max_lim_prod, max_lim_prod))

    for i, j, k in itertools.product(range(max_lim_1), repeat=3):
        for i2, j2, k2 in itertools.product(range(max_lim_2), repeat=3):
            poly_coeff_prod[i + i2, j + j2, k + k2] += poly_coeff[i, j, k] * poly_coeff2[i2, j2, k2]

    if max_lim is not None:
        poly_coeff_prod = poly_coeff_prod[:max_lim, :max_lim, :max_lim]

    return poly_coeff_prod


def exp_poly_coeff(poly_coeff, k, max_lim=None):
    """
    Calculate the exponential of a polynomial coefficient matrix

    :param poly_coeff: polynomial coefficient matrix
    :param k: exponent
    :param max_lim:
    :return:
    """

    max_lim_x = len(poly_coeff) * max(k, 1)

    poly_coeff_exp = np.zeros((max_lim_x, max_lim_x, max_lim_x))
    poly_coeff_exp[0, 0, 0] = 1
    for _ in range(k):
        poly_coeff_exp = product_poly_coeff(poly_coeff_exp, poly_coeff, max_lim=max_lim)

    return poly_coeff_exp


def gaussian_product(g_1, g_2):
    """
    Calculate the product of two Gaussian primitives. (s functions only)
    """
    alpha = g_1.alpha + g_2.alpha
    prefactor = g_1.prefactor * g_2.prefactor
    p = g_1.alpha * g_2.alpha

    PG = g_1.coordinates - g_2.coordinates
    coordinates = (g_1.coordinates * g_1.alpha + g_2.coordinates * g_2.alpha) / alpha
    K = np.exp(-p/alpha * np.dot(PG, PG))

    poly_coeff = product_poly_coeff(g_1.poly_coeff, g_2.poly_coeff)

    return PrimitiveGaussian(alpha, K * prefactor, coordinates, normalize=False, poly_coeff=poly_coeff)


def integrate_exponential_simple_py(n, a):
    """
    integrals type   x^n exp(-ax^2)

    """
    if n == 0:
        return np.sqrt(np.pi/a)
    elif np.mod(n, 2) == 0:
        k = n//2
        return 2*np.math.factorial(np.math.factorial(2*k-1))/(2**(k+1)*a**k)*np.sqrt(np.pi/a)
    else:
        return 0.0


def integrate_exponential_py(n, a, b):
    """
    integrals type  x^n exp(-(ax^2+bx))

    """
    if n == 0:
        return np.sqrt(np.pi/a)*np.exp(b**2/(4*a))
    elif n == 1:
        return np.sqrt(np.pi)/(2*a**(3/2))*b*np.exp(b**2/(4*a))
    else:
        factor = np.sum([math.comb(n, 2*k)*(b/(2*a))**(n-2*k)*math.factorial(2*k)/(2**(2*k)*math.factorial(k)*a**k)
                          for k in range(n//2+1)])
        return factor * np.sqrt(np.pi/a)*np.exp(b**2/(4*a))


def gaussian_integral_py(alpha, center, poly_coeff):
    max_lim = len(poly_coeff)
    pre_exponential = np.exp(-alpha * np.dot(center, center))

    integrate = 0.0
    for i, j, k in itertools.product(range(max_lim), repeat=3):
        integrate += poly_coeff[i, j, k] * np.prod([integrate_exponential_py(l, alpha, 2 * alpha * c) for c, l in zip(center, [i, j, k])])

    return pre_exponential * integrate


class PrimitiveGaussian:
    def __init__(self, alpha, prefactor=1.0, coordinates=(0, 0, 0), l=(0, 0, 0), normalize=True, poly_coeff=None):
        self._n_dim = len(coordinates)
        self.alpha = alpha
        self.prefactor = prefactor
        self.coordinates = np.array(coordinates)
        self.l = l
        if poly_coeff is None:
            self.poly_coeff = binomial_expansion(l, -self.coordinates)
        else:
            self.poly_coeff = poly_coeff

        # normalize primitive such that <prim|prim> = 1
        if normalize:
            norm = self._get_norm()

            self.prefactor = prefactor / np.sqrt(norm)

    @property
    def integrate(self):
        return self.prefactor * gaussian_integral(self.alpha, self.coordinates, self.poly_coeff)

    def _get_norm(self):
        poly_coeff_sq = product_poly_coeff(self.poly_coeff, self.poly_coeff)
        return self.prefactor * gaussian_integral(2*self.alpha, self.coordinates, poly_coeff_sq)

    def __call__(self, value):
        value = np.array(value)
        max_lim = len(self.poly_coeff)

        coef_matrix = np.fromfunction(lambda i, j, k: value[0]**i*value[1]**j*value[2]**k, (max_lim, max_lim, max_lim))
        angular = np.sum(self.poly_coeff * coef_matrix)

        return self.prefactor * angular * np.exp(-self.alpha * np.linalg.norm(value - self.coordinates)**2)

    def __mul__(self, other):
        return gaussian_product(self, other)

    def apply_translation(self, translation):
        translation = np.array(translation)
        max_lim = len(self.poly_coeff)
        poly_coeff_trans = np.zeros_like(self.poly_coeff)

        for i, j, k in itertools.product(range(max_lim), repeat=3):
            poly_coeff_trans += self.poly_coeff[i, j, k] * binomial_expansion([i, j, k], -translation, max_lim=max_lim)
        self.coordinates = self.coordinates + translation
        self.poly_coeff = poly_coeff_trans

    def apply_rotation(self, angle, axis):

        from posym.operations.rotation import rotation
        rot_matrix = rotation(angle, axis)
        max_lim = len(self.poly_coeff)

        if max_lim > 1:
            max_lim_final = max_lim * 3 # * NDIM (naive max dimmension can be improved)
            poly_coeff_rot = np.zeros((max_lim_final, max_lim_final, max_lim_final))

            for i, j, k in itertools.product(range(max_lim), repeat=3):
                poly_temp_xyz = np.zeros((max_lim_final, max_lim_final, max_lim_final))
                poly_temp_xyz[0, 0, 0] = 1.0
                for l, m in enumerate([i, j, k]):
                    poly_temp = np.zeros((max_lim, max_lim, max_lim))
                    poly_temp[1, 0, 0] = rot_matrix[0, l]
                    poly_temp[0, 1, 0] = rot_matrix[1, l]
                    poly_temp[0, 0, 1] = rot_matrix[2, l]
                    poly_temp = exp_poly_coeff(poly_temp, m, max_lim_final)

                    poly_temp_xyz = product_poly_coeff(poly_temp_xyz, poly_temp, max_lim_final)

                poly_coeff_rot += self.poly_coeff[i, j, k] * poly_temp_xyz

            self.poly_coeff = poly_coeff_rot
        self.coordinates = np.dot(rot_matrix, self.coordinates)


class BasisFunction:
    def __init__(self, primitive_gaussians, coefficients, coordinates=None):
        primitive_gaussians = deepcopy(primitive_gaussians)
        if coordinates is not None:
            coordinates = np.array(coordinates)
            for primitive in primitive_gaussians:
                primitive.apply_translation(coordinates - primitive.coordinates)

        self.primitive_gaussians = primitive_gaussians
        self.coefficients = coefficients

    def get_number_of_primitives(self):
        return len(self.primitive_gaussians)

    def set_coordinates(self, coordinates):
        for primitive in self.primitive_gaussians:
            primitive.coordinates = np.array(coordinates)

    def apply_translation(self, translation):
        primitive_gaussians = deepcopy(self.primitive_gaussians)
        for primitive in primitive_gaussians:
            primitive.apply_translation(translation)

        self.primitive_gaussians = primitive_gaussians

    def apply_rotation(self, angle, axis):
        primitive_gaussians = deepcopy(self.primitive_gaussians)
        for primitive in primitive_gaussians:
            primitive.apply_rotation(angle, axis)

        self.primitive_gaussians = primitive_gaussians

    @property
    def integrate(self):
        return sum([coef * prim.integrate for coef, prim in zip(self.coefficients, self.primitive_gaussians)])

    def __call__(self, value):
        return sum([coef * prim(value) for coef, prim in zip(self.coefficients, self.primitive_gaussians)])

    def __mul__(self, other):
        if isinstance(other, float):
            return BasisFunction(self.primitive_gaussians, [coef * other for coef in self.coefficients])

        elif isinstance(other, BasisFunction):

            primitive_gaussians = []
            coefficients = []

            for primitive_1, coeff_1 in zip(self.primitive_gaussians, self.coefficients):
                for primitive_2, coeff_2 in zip(other.primitive_gaussians, other.coefficients):
                    primitive_gaussians.append(primitive_1 * primitive_2)
                    coefficients.append(coeff_1 * coeff_2)

            return BasisFunction(primitive_gaussians, coefficients)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return BasisFunction(self.primitive_gaussians + other.primitive_gaussians,
                             self.coefficients + other.coefficients)

    def __sub__(self, other):
        negative_coefficients = list(-np.array(other.coefficients))
        return BasisFunction(self.primitive_gaussians + other.primitive_gaussians,
                             self.coefficients + negative_coefficients)


if __name__ == '__main__':

    # test Lithium
    """
    Li     0
    S    3   1.00
          0.1611957475D+02       0.1543289673D+00
          0.2936200663D+01       0.5353281423D+00
          0.7946504870D+00       0.4446345422D+00
    SP   3   1.00
          0.6362897469D+00      -0.9996722919D-01       0.1559162750D+00
          0.1478600533D+00       0.3995128261D+00       0.6076837186D+00
          0.4808867840D-01       0.7001154689D+00       0.3919573931D+00
    ****
3   d
      0.2145684671D+02       0.2197679508D+00
      0.6545022156D+01       0.6555473627D+00
      0.2525273021D+01       0.2865732590D+00
*
    """

    sa = PrimitiveGaussian(alpha=16.11957475, l=[0, 0, 0], coordinates=(0.0, 0.0, 0.0), normalize=True)
    sb = PrimitiveGaussian(alpha=2.936200663, l=[0, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    sc = PrimitiveGaussian(alpha=0.794650487, l=[0, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    s1 = BasisFunction([sa, sb, sc], [0.1543289673, 0.5353281423, 0.4446345422])
    print('s:', (s1*s1).integrate)
    s1.apply_rotation(0.33*2*np.pi/2, [0.2, 5.0, 1.0])

    s2a = PrimitiveGaussian(alpha=0.6362897469, l=[0, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    s2b = PrimitiveGaussian(alpha=0.1478600533, l=[0, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    s2c = PrimitiveGaussian(alpha=0.0480886784, l=[0, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    s2 = BasisFunction([s2a, s2b, s2c], [-0.09996722919, 0.3995128261, 0.7001154689])
    print('s2:', (s2*s2).integrate)

    pxa = PrimitiveGaussian(alpha=0.6362897469, l=[1, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pxb = PrimitiveGaussian(alpha=0.1478600533, l=[1, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pxc = PrimitiveGaussian(alpha=0.0480886784, l=[1, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    px = BasisFunction([pxa, pxb, pxc], [0.1559162750, 0.6076837186, 0.3919573931],
                       coordinates=[0.0, 0.0, 0.0]
                       )
    print('px:', (px*px).integrate)
    px.apply_rotation(np.pi/2, [0.0, 0.0, 1.0])

    print('px rot:', (px*px).integrate)

    print('pxa:', (pxa*pxa).integrate)
    pxa.apply_rotation(2*np.pi/2, [0.0, 0.0, 1.0])
    print('pxa:', (pxa*pxa).integrate)


    o1 = -0.992527759 * s1 -0.0293095626 * s2
    print('o1:', (o1*o1).integrate)

    o2 = 0.276812804 * s1 -1.02998914 * s2
    print('o2:', (o2*o2).integrate)

    pya = PrimitiveGaussian(alpha=0.6362897469, l=[0, 1, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pyb = PrimitiveGaussian(alpha=0.1478600533, l=[0, 1, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pyc = PrimitiveGaussian(alpha=0.0480886784, l=[0, 1, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    py = BasisFunction([pya, pyb, pyc], [0.1559162750, 0.6076837186, 0.3919573931])
    print('py:', (py*py).integrate)

    pza = PrimitiveGaussian(alpha=0.6362897469, l=[0, 0, 1], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pzb = PrimitiveGaussian(alpha=0.1478600533, l=[0, 0, 1], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pzc = PrimitiveGaussian(alpha=0.0480886784, l=[0, 0, 1], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pz = BasisFunction([pza, pzb, pzc], [0.1559162750, 0.6076837186, 0.3919573931])
    print('pz:', (pz*pz).integrate)

    d1a = PrimitiveGaussian(alpha=21.45684671, l=[1, 1, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    d1b = PrimitiveGaussian(alpha=6.545022156, l=[1, 1, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    d1c = PrimitiveGaussian(alpha=2.525273021, l=[1, 1, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    d1 = BasisFunction([d1a, d1b, d1c], [0.2197679508, 0.6555473627, 0.2865732590])
    d1.apply_rotation(np.pi/4, [0.0, 0.0, 1.0])
    print('d1:', (d1*d1).integrate)

    px2 = px * px
    print('px2:', px2.integrate)

    # from scipy import integrate
    # f = lambda x, y, z: px2([x, y, z])
    # num_integral = integrate.tplquad(f, -5, 5, lambda x: -5, lambda x: 5, lambda x, y: -5, lambda x, y: 5)
    # print('num_integral px*px', num_integral)

    import matplotlib.pyplot as plt
    x = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, x)
    Z = []
    for x, y in zip(X, Y):
        zv = [d1([x2, y2, 0.0]) for x2, y2 in zip(x, y)]
        Z.append(zv)
    Z = np.array(Z)

    plt.contour(X, Y, Z, colors='k')
    plt.show()

    x = np.linspace(-5, 5, 200)
    plt.plot(x, [o1([x_,  0.0, 0.0]) for x_ in x], label='O1')
    plt.plot(x, [o2([x_,  0.0, 0.0]) for x_ in x], label='O2')
    plt.plot(x, [px([x_,  0.0, 0.0])*py([x_, 0, 0])for x_ in x], label='px*py')
    plt.plot(x, [px2([x_, 0.0, 0.0]) for x_ in x], '--', label='px2')
    plt.plot(x, [d1([x_,  0.0, 0.0]) for x_ in x], '-', label='d1')
    plt.plot(x, [pxa([x_, 0.0, 0.0]) for x_ in x], '-', label='pxa')

    plt.legend()
    plt.show()


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

    mo_coefficients = [[ 0.994216442, 0.025846814, 0.000000000, 0.000000000,-0.004164076,-0.005583712, -0.005583712],
                       [ 0.233766661,-0.844456594, 0.000000000, 0.000000000, 0.122829781,-0.155593214, -0.155593214],
                       [ 0.000000000, 0.000000000, 0.612692349, 0.000000000, 0.000000000,-0.449221684,  0.449221684],
                       [-0.104033343, 0.538153649, 0.000000000, 0.000000000, 0.755880259,-0.295107107, -0.295107107],
                       [ 0.000000000, 0.000000000, 0.000000000,-1.000000000, 0.000000000, 0.000000000,  0.000000000],
                       [-0.125818566, 0.820120983, 0.000000000, 0.000000000,-0.763538862,-0.769155124, -0.769155124],
                       [ 0.000000000, 0.000000000, 0.959800163, 0.000000000, 0.000000000, 0.814629717, -0.814629717]]

    coordinates=[[ 0.0000000000, 0.000000000, -0.0428008531],
                 [-0.7581074140, 0.000000000, -0.6785995734],
                 [ 0.7581074140, 0.000000000, -0.6785995734]]

    # Oxigen atom
    sa = PrimitiveGaussian(alpha=130.70932)
    sb = PrimitiveGaussian(alpha=23.808861)
    sc = PrimitiveGaussian(alpha=6.4436083)
    s_O = BasisFunction([sa, sb, sc],
                        [0.154328969, 0.535328136, 0.444634536],
                        coordinates=[0.0000000000, 0.000000000, -0.0808819]) # Bohr

    sa = PrimitiveGaussian(alpha=5.03315132)
    sb = PrimitiveGaussian(alpha=1.1695961)
    sc = PrimitiveGaussian(alpha=0.3803890)
    s2_O = BasisFunction([sa, sb, sc],
                         [-0.099967228, 0.399512825, 0.700115461],
                         coordinates=[0.0000000000, 0.000000000,  -0.0808819])

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
                         coordinates=[0.0000000000, 0.000000000,  -0.0808819])
    py_O = BasisFunction([pya, pyb, pyc],
                         [0.155916268, 0.6076837186, 0.3919573931],
                         coordinates=[0.0000000000, 0.000000000,  -0.0808819])
    pz_O = BasisFunction([pza, pzb, pzc],
                         [0.155916268, 0.6076837186, 0.3919573931],
                         coordinates=[0.0000000000, 0.000000000,  -0.0808819])

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

    o1 = s_O * 0.994216442 + s2_O * 0.025846814 + px_O * 0.0 + py_O * 0.0 + pz_O * \
         -0.004164076 + s_H * -0.005583712 + s2_H * -0.005583712

    o2 = s_O * 0.0 + s2_O * 0.0 + px_O * 0.612692349 + py_O * 0.0 + pz_O * 0.0 + s_H * \
         -0.44922168 + s2_H * 0.449221684

    print('dot o1o1', (o1*o1).integrate)
    print('dot o2o2', (o2*o2).integrate)
    print('dot o1o2', (o2*o1).integrate)

    density_matrix = 2 * np.outer(mo_coefficients[0], mo_coefficients[0]) + \
                     2 * np.outer(mo_coefficients[1], mo_coefficients[1]) + \
                     2 * np.outer(mo_coefficients[2], mo_coefficients[2]) + \
                     2 * np.outer(mo_coefficients[3], mo_coefficients[3]) + \
                     0 * np.outer(mo_coefficients[4], mo_coefficients[4]) + \
                     0 * np.outer(mo_coefficients[5], mo_coefficients[5]) + \
                     0 * np.outer(mo_coefficients[6], mo_coefficients[6])

    print('density matrix\n', density_matrix[:4, :4])

    basis_functions = [s_O, s2_O, px_O, py_O, pz_O, s_H, s2_H]
    total_electrons = 0
    for i, bf1 in enumerate(basis_functions):
        for j, bf2 in enumerate(basis_functions):
            total_electrons += (bf1*bf2).integrate * density_matrix[i, j]


    print('total electrons', total_electrons)

    def get_overlap_density_naive(basis_set_1, basis_set_2, density_matrix):
        n = len(basis_set_1)
        s = np.zeros((n, n, n, n))

        for i, basis1 in enumerate(basis_set_1):
            for j, basis2 in enumerate(basis_set_1):
                for k, basis3 in enumerate(basis_set_2):
                    for l, basis4 in enumerate(basis_set_2):
                        dens_prod = density_matrix[i, j] * density_matrix[k, l]
                        basis_prod = basis1 * basis2 * basis3 * basis4
                        s[i, j, k, l] = basis_prod.integrate * dens_prod

        return np.sum(s)

    from sympy.utilities.iterables import multiset_permutations
    def get_overlap_density(basis_set_1, basis_set_2, density_matrix):
        n = len(basis_set_1)
        s = np.zeros((n, n, n, n))

        for i in range(n):
            for j in range(i+1):
                for k in range(n):
                    for l in range(k+1):
                        dens_prod = density_matrix[i, j] * density_matrix[k, l]
                        integral = (basis_set_1[i] * basis_set_1[j] * basis_set_2[k] * basis_set_2[l]).integrate
                        for perm in multiset_permutations([i, j]):
                            for perm2 in multiset_permutations([k, l]):
                                s[perm[0], perm[1], perm2[0], perm2[1]] = integral * dens_prod

        return np.sum(s)

    def get_self_similarity(basis_set_1, density_matrix):
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

    self_similarity = get_overlap_density(basis_functions, basis_functions, density_matrix)
    print('self_similarity', self_similarity)

    self_similarity = get_self_similarity(basis_functions, density_matrix)
    print('self_similarity', self_similarity)

    def rotate_basis_set(basis_set, angle, axis):
        import copy
        new_basis_set = copy.deepcopy(basis_set)
        for bf in new_basis_set:
            bf.apply_rotation(angle, axis)
        return new_basis_set

    def translate_basis_set(basis_set, translation):
        import copy
        new_basis_set = copy.deepcopy(basis_set)
        for bf in new_basis_set:
            bf.apply_translation(translation)
        return new_basis_set


    basis_functions_r = rotate_basis_set(basis_functions, np.pi, [1, 0, 0])
    basis_functions_t = translate_basis_set(basis_functions, [1, 0, 0])

    s_O, s2_O, px_O, py_O, pz_O, s_H, s2_H = basis_functions_r
    o2 = s_O * 0.0 + s2_O * 0.0 + px_O * 0.612692349 + py_O * 0.0 + pz_O * 0.0 + s_H * -0.44922168 + s2_H * 0.449221684
    print('dot(rot)', (o2*o2).integrate)

    s_O, s2_O, px_O, py_O, pz_O, s_H, s2_H = basis_functions_t
    o2 = s_O * 0.0 + s2_O * 0.0 + px_O * 0.612692349 + py_O * 0.0 + pz_O * 0.0 + s_H * -0.44922168 + s2_H * 0.449221684
    print('dot(trans)', (o2*o2).integrate)

    print('measure dens:', get_overlap_density(basis_functions, basis_functions_r, density_matrix)/self_similarity)


    # Fock matrix
    fock_matrix = np.array([[-20.24290797856165, -5.163857371773464, 0.0, 0.0, 0.028693754710997377, -1.1083321876539016, -1.1083321876539016],
                   [-5.163857371773464, -2.4396904148844447, 0.0, 0.0, 0.11807009938463243, -0.9717587486764925, -0.9717587486764925],
                   [ 0.0, 0.0, -0.2938600995387204, 0.0, 0.0, 0.37840779313794853, -0.37840779313794853],
                   [ 0.0, 0.0, 0.0, -0.392614627464097, 0.0, -2.4402816249492116e-17, 2.4402816249492126e-17],
                   [ 0.028693754710997377, 0.11807009938463243, 0.0, 0.0, -0.33633212331400564, 0.37348029921997167, 0.3734802992199712],
                   [-1.1083321876539016, -0.9717587486764925, 0.37840779313794853, -2.4402816249492116e-17, 0.37348029921997167, -0.5411484908071187, -0.3753070759124842],
                   [-1.1083321876539016, -0.9717587486764925, -0.37840779313794853, 2.4402816249492126e-17, 0.3734802992199712, -0.3753070759124842, -0.5411484908071187]])

    self_similarity = get_self_similarity(basis_functions, fock_matrix)
    print('measure fock:', get_overlap_density(basis_functions, basis_functions_r, fock_matrix)/self_similarity)
