import numpy as np
from copy import deepcopy
import math
import itertools


def binomial_transformation(l, x, max_lim=None):

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

def product_poly_coeff(poly_coeff, poly_coeff2, max_lim=None):
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


def integrate_exponential_simple(n, a):
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


def integrate_exponential(n, a, b):
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
        # factor = np.sum([(b/np.sqrt(a))**(n-2*k)/(math.factorial(n)*math.factorial(n-2*k)) for k in range(n//2+1)])
        # return factor*np.sqrt(np.pi/a)*np.exp(b**2/(4*a))*math.factorial(n)**2*(1/(2*np.sqrt(a)))**n

class PrimitiveGaussian:
    def __init__(self, alpha, prefactor=1.0, coordinates=(0, 0, 0), l=(0, 0, 0), normalize=True, poly_coeff=None):
        self._n_dim = len(coordinates)
        self.alpha = alpha
        self.prefactor = prefactor
        self.coordinates = np.array(coordinates)
        self.l = l
        if poly_coeff is None:
            self.poly_coeff = binomial_transformation(l, -self.coordinates)
        else:
            self.poly_coeff = poly_coeff

        # normalize primitive such that <prim|prim> = 1
        if normalize:
            # norm = prefactor * (np.pi/(2*self.alpha))**(self._n_dim/2)
            norm = self._get_norm()

            self.prefactor = prefactor / np.sqrt(norm)

        # self.integrate_old = self.prefactor * (np.pi / self.alpha)**(self._n_dim/2)

    @property
    def integrate(self):
        max_lim = len(self.poly_coeff)
        integrate = 0.0
        pre_exponential = np.exp(-self.alpha * np.dot(self.coordinates, self.coordinates))
        for i, j, k in itertools.product(range(max_lim), repeat=3):
            # integrate += self.poly_coeff[i, j, k] * np.prod([integrate_exponential_simple(l, self.alpha) for l in [i, j, k]])
            integrate += self.poly_coeff[i, j, k] * np.prod([integrate_exponential(l, self.alpha, 2 * self.alpha * c) for c, l in zip(self.coordinates, [i, j, k])])

        return integrate * self.prefactor * pre_exponential

    def _get_norm(self):
        poly_coeff_sq = product_poly_coeff(self.poly_coeff, self.poly_coeff)
        pre_exponential = np.exp(-2*self.alpha * np.dot(self.coordinates, self.coordinates))
        max_lim = len(poly_coeff_sq)

        integrate = 0.0
        for i, j, k in itertools.product(range(max_lim), repeat=3):
            # integrate += poly_coeff_sq[i, j, k] * np.prod([integrate_exponential_simple(l, 2*self.alpha) for l in [i, j, k]])
            integrate += poly_coeff_sq[i, j, k] * np.prod([integrate_exponential(l, 2*self.alpha, 4 * self.alpha * c) for c, l in zip(self.coordinates, [i, j, k])])

        return integrate * self.prefactor * pre_exponential

    def __call__(self, value):
        value = np.array(value)
        max_lim = len(self.poly_coeff)

        #angular = 0.0
        #for i in range(4):
        #    for j in range(4):
        #        for k in range(4):
        #            if self.poly_coeff[i, j, k]:
        #                angular += self.poly_coeff[i, j, k] * np.prod([(value[m])**l for m, l in enumerate([i, j, k])])

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
            poly_coeff_trans += self.poly_coeff[i, j, k] * binomial_transformation([i, j, k], -translation, max_lim=max_lim)
        self.coordinates += translation

        self.poly_coeff = poly_coeff_trans


class BasisFunction:
    def __init__(self, primitive_gaussians, coefficients, coordinates=None):
        primitive_gaussians = deepcopy(primitive_gaussians)
        if coordinates is not None:
            for primitive in primitive_gaussians:
                primitive.apply_translation(coordinates - primitive.coordinates)
                # primitive.coordinates = np.array(coordinates)
                # raise Exception("Not fully implemented yet")

        self.primitive_gaussians = primitive_gaussians
        self.coefficients = coefficients

    def get_number_of_primitives(self):
        return len(self.primitive_gaussians)

    def set_coordinates(self, coordinates):
        for primitive in self.primitive_gaussians:
            primitive.coordinates = np.array(coordinates)

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

    center = [1.0, 0.0, 0.0]
    sa = PrimitiveGaussian(alpha=16.11957475, l=[0, 0, 0], coordinates=(0.0, 0.0, 0.0), normalize=True)
    sb = PrimitiveGaussian(alpha=2.936200663, l=[0, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    sc = PrimitiveGaussian(alpha=0.794650487, l=[0, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    s1 = BasisFunction([sa, sb, sc], [0.1543289673, 0.5353281423, 0.4446345422])
    print('s:', (s1*s1).integrate)

    s2a = PrimitiveGaussian(alpha=0.6362897469, l=[0, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    s2b = PrimitiveGaussian(alpha=0.1478600533, l=[0, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    s2c = PrimitiveGaussian(alpha=0.04808867840, l=[0, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    s2 = BasisFunction([s2a, s2b, s2c], [-0.09996722919, 0.3995128261, 0.7001154689])
    print('s2:', (s2*s2).integrate)

    pxa = PrimitiveGaussian(alpha=0.6362897469, l=[1, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pxb = PrimitiveGaussian(alpha=0.1478600533, l=[1, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pxc = PrimitiveGaussian(alpha=0.04808867840, l=[1, 0, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    px = BasisFunction([pxa, pxb, pxc], [0.1559162750, 0.6076837186, 0.3919573931],
                       coordinates=[1.0, 0.0, 0.0]
                       )
    print('px:', (px*px).integrate)

    o1 = -0.992527759 * s1 -0.0293095626 * s2
    print('o1:', (o1*o1).integrate)

    o2 = 0.276812804 * s1 -1.02998914 * s2
    print('o2:', (o2*o2).integrate)

    pya = PrimitiveGaussian(alpha=0.6362897469, l=[0, 1, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pyb = PrimitiveGaussian(alpha=0.1478600533, l=[0, 1, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pyc = PrimitiveGaussian(alpha=0.04808867840, l=[0, 1, 0], coordinates=[0.0, 0.0, 0.0], normalize=True)
    py = BasisFunction([pya, pyb, pyc], [0.1559162750, 0.6076837186, 0.3919573931])
    print('py:', (py*py).integrate)

    pza = PrimitiveGaussian(alpha=0.6362897469, l=[0, 0, 1], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pzb = PrimitiveGaussian(alpha=0.1478600533, l=[0, 0, 1], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pzc = PrimitiveGaussian(alpha=0.04808867840, l=[0, 0, 1], coordinates=[0.0, 0.0, 0.0], normalize=True)
    pz = BasisFunction([pza, pzb, pzc], [0.1559162750, 0.6076837186, 0.3919573931])
    print('pz:', (pz*pz).integrate)

    d1a = PrimitiveGaussian(alpha=21.45684671, l=[2, 0, 0], coordinates=[2.0, 0.0, 0.0], normalize=True)
    d1b = PrimitiveGaussian(alpha=6.545022156, l=[2, 0, 0], coordinates=[2.0, 0.0, 0.0], normalize=True)
    d1c = PrimitiveGaussian(alpha=2.525273021, l=[2, 0, 0], coordinates=[2.0, 0.0, 0.0], normalize=True)
    d1 = BasisFunction([d1a, d1b, d1c], [0.2197679508, 0.6555473627, 0.2865732590])
    print('d1:', (d1*d1).integrate)

    px2 = px * px
    # from scipy import integrate
    # f = lambda x, y, z: px2([x, y, z])
    # num_integral = integrate.tplquad(f, -5, 5, lambda x: -5, lambda x: 5, lambda x, y: -5, lambda x, y: 5)
    # print('num_integral px*px', num_integral)

    import matplotlib.pyplot as plt
    x = np.linspace(-5, 5, 500)

    plt.plot(x, [o1([x_, 0.1, 0]) for x_ in x])

    plt.plot(x, [o2([x_, 0.1, 0]) for x_ in x])
    plt.plot(x, [px([x_, 0, 0])*py([x_, 0, 0])for x_ in x])
    plt.plot(x, [px2([x_, 0.1, 0]) for x_ in x], '--')
    plt.plot(x, [d1([x_, 0.0, 0]) for x_ in x], '-')

    plt.show()
