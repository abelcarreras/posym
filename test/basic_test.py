from posym import PointGroup, SymmetryBase, SymmetryFunction, SymmetryModes
import unittest
from posym.algebra import dot, norm
import numpy as np


class OperationsTest(unittest.TestCase):

    def test_pg(self):

        pg = PointGroup(group='Td')

        self.assertCountEqual(pg.op_labels, ['E', 'C3', 'C2', 'S4', 'sd'])
        self.assertCountEqual(pg.ir_labels, ['A1', 'A2', 'E', 'T1', 'T2'])
        self.assertCountEqual(pg.ir_degeneracies, [1, 1, 2, 3, 3])
        self.assertEqual(pg.order, 24)

    def test_algebra(self):

        a1 = SymmetryBase(group='Td', rep='A1')
        a2 = SymmetryBase(group='Td', rep='A2')
        e = SymmetryBase(group='Td', rep='E')
        t1 = SymmetryBase(group='Td', rep='T1')
        t2 = SymmetryBase(group='Td', rep='T2')

        prod_1 = e * e + 2 * a1
        result_1 = 3*a1 + a2 + e

        prod_2 = t1*t1
        result_2 = a1 + e + t1 + t2

        self.assertCountEqual(prod_1.get_ir_representation(), result_1.get_ir_representation())
        self.assertCountEqual(prod_2.get_ir_representation(), result_2.get_ir_representation())

    def test_functions(self):

        a1 = SymmetryBase(group='Td', rep='A1')
        a2 = SymmetryBase(group='Td', rep='A2')
        e = SymmetryBase(group='Td', rep='E')
        t1 = SymmetryBase(group='Td', rep='T1')
        t2 = SymmetryBase(group='Td', rep='T2')

        print('A1 . A1: ', dot(a1, a1))

        self.assertAlmostEqual(dot(e, e), 4)
        self.assertAlmostEqual(dot(t1, e), 0)
        self.assertAlmostEqual(dot(t1, t1 + e), 9.0)
        self.assertAlmostEqual(dot(t1, t1 + e, normalize=True), 0.6)
        self.assertAlmostEqual(dot(t1 + e, t1 + e), 25.0)
        self.assertAlmostEqual(dot(t1 + e, t1 + e, normalize=True), 1.0)
        self.assertAlmostEqual(dot(0.6 * t1 + e, t1), 5.4)
        self.assertAlmostEqual(dot(t1, t1, normalize=True), 1.0)
        self.assertAlmostEqual(norm(t1 + e), 5.0)
        self.assertAlmostEqual(norm(t1), 3.0)
        self.assertAlmostEqual(norm(e), 2.0)
        self.assertAlmostEqual(norm(a1), 1.0)
        self.assertAlmostEqual(norm(a1 + a2 + e + t1 + t2), 10.0)

class BasisTest(unittest.TestCase):

    def setUp(self):

        from posym.basis import PrimitiveGaussian, BasisFunction

        self.sa = PrimitiveGaussian(alpha=16.11957475, l=[0, 0, 0], center=(0.0, 0.0, 0.0), normalize=True)
        self.sb = PrimitiveGaussian(alpha=2.936200663, l=[0, 0, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.sc = PrimitiveGaussian(alpha=0.794650487, l=[0, 0, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.s1 = BasisFunction([self.sa, self.sb, self.sc], [0.1543289673, 0.5353281423, 0.4446345422])

        self.s2a = PrimitiveGaussian(alpha=0.6362897469, l=[0, 0, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.s2b = PrimitiveGaussian(alpha=0.1478600533, l=[0, 0, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.s2c = PrimitiveGaussian(alpha=0.0480886784, l=[0, 0, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.s2 = BasisFunction([self.s2a, self.s2b, self.s2c], [-0.09996722919, 0.3995128261, 0.7001154689])

        self.pxa = PrimitiveGaussian(alpha=0.6362897469, l=[1, 0, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.pxb = PrimitiveGaussian(alpha=0.1478600533, l=[1, 0, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.pxc = PrimitiveGaussian(alpha=0.0480886784, l=[1, 0, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.px = BasisFunction([self.pxa, self.pxb, self.pxc],
                                [0.1559162750, 0.6076837186, 0.3919573931],
                                center=[0.0, 0.0, 0.0]
                                )

        self.pya = PrimitiveGaussian(alpha=0.6362897469, l=[0, 1, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.pyb = PrimitiveGaussian(alpha=0.1478600533, l=[0, 1, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.pyc = PrimitiveGaussian(alpha=0.0480886784, l=[0, 1, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.py = BasisFunction([self.pya, self.pyb, self.pyc], [0.1559162750, 0.6076837186, 0.3919573931])

        self.pza = PrimitiveGaussian(alpha=0.6362897469, l=[0, 0, 1], center=[0.0, 0.0, 0.0], normalize=True)
        self.pzb = PrimitiveGaussian(alpha=0.1478600533, l=[0, 0, 1], center=[0.0, 0.0, 0.0], normalize=True)
        self.pzc = PrimitiveGaussian(alpha=0.0480886784, l=[0, 0, 1], center=[0.0, 0.0, 0.0], normalize=True)
        self.pz = BasisFunction([self.pza, self.pzb, self.pzc], [0.1559162750, 0.6076837186, 0.3919573931])

        self.d1a = PrimitiveGaussian(alpha=21.45684671, l=[1, 1, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.d1b = PrimitiveGaussian(alpha=6.545022156, l=[1, 1, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.d1c = PrimitiveGaussian(alpha=2.525273021, l=[1, 1, 0], center=[0.0, 0.0, 0.0], normalize=True)
        self.d1 = BasisFunction([self.d1a, self.d1b, self.d1c], [0.2197679508, 0.6555473627, 0.2865732590])

        pass

    def test_s_contractions(self):
        self.assertAlmostEqual((self.sa * self.sa).integrate, 1)
        self.assertAlmostEqual((self.sb * self.sb).integrate, 1)
        self.assertAlmostEqual((self.sc * self.sc).integrate, 1)
        self.assertAlmostEqual((self.s1 * self.s1).integrate, 1)

    def test_s_linear_operations(self):
        s1 = self.s1.copy()
        s1.apply_rotation(0.33 * 2 * np.pi / 2, [0.2, 5.0, 1.0])
        s1.apply_translation([-0.1, 5.0, -1.0])
        self.assertAlmostEqual((s1 * s1).integrate, 1)

        s2 = self.s2.copy()
        s2.apply_translation([0.1, 0.1, 5.0])
        s2.apply_rotation(0.88 * 2 * np.pi / 2, [0.2, -0.6, 1.0])
        self.assertAlmostEqual((s2 * s2).integrate, 1)

    def test_p_contractions(self):

        self.assertAlmostEqual((self.pxa * self.pxa).integrate, 1)
        self.assertAlmostEqual((self.pxb * self.pxb).integrate, 1)
        self.assertAlmostEqual((self.pxc * self.pxc).integrate, 1)

        self.assertAlmostEqual((self.px * self.px).integrate, 1)
        self.assertAlmostEqual((self.py * self.py).integrate, 1)
        self.assertAlmostEqual((self.pz * self.pz).integrate, 1)

    def test_p_linear_operations(self):
        px = self.px.copy()
        px.apply_rotation(0.33 * 2 * np.pi / 2, [0.2, 5.0, 1.0])
        px.apply_translation([-0.1, 5.0, -1.0])
        self.assertAlmostEqual((px * px).integrate, 1)

        py = self.py.copy()
        py.apply_rotation(0.33 * 2 * np.pi / 2, [0.2, 5.0, 1.0])
        py.apply_translation([-0.1, 5.0, -1.0])
        self.assertAlmostEqual((py * py).integrate, 1)

        pz = self.pz.copy()
        pz.apply_rotation(0.33 * 2 * np.pi / 2, [0.2, 5.0, 1.0])
        pz.apply_translation([-0.1, 5.0, -1.0])
        self.assertAlmostEqual((pz * pz).integrate, 1)

    def test_d_contractions(self):

        self.assertAlmostEqual((self.d1a * self.d1a).integrate, 1)
        self.assertAlmostEqual((self.d1b * self.d1b).integrate, 1)
        self.assertAlmostEqual((self.d1c * self.d1c).integrate, 1)

        self.assertAlmostEqual((self.d1 * self.d1).integrate, 1)

    def test_d_linear_operations(self):
        d1 = self.px.copy()
        d1.apply_rotation(0.33 * 2 * np.pi / 2, [0.2, 5.0, 1.0])
        self.assertAlmostEqual((d1 * d1).integrate, 1)

        d1.apply_translation([-0.1, 5.0, -1.0])
        self.assertAlmostEqual((d1 * d1).integrate, 1)

    def test_cross_products(self):

        for f1 in [self.s1, self.px, self.py, self.pz, self.d1]:
            for f2 in [self.s1, self.px, self.py, self.pz, self.d1]:
                if f1 == f2:
                    self.assertAlmostEqual((f1 * f1).integrate, 1)
                else:
                    self.assertAlmostEqual((f1 * f2).integrate, 0)

    def test_molecular_orbitals_s(self):
        o1 = -0.992527759 * self.s1 - 0.0293095626 * self.s2
        print('o1:', (o1 * o1).integrate)

        o2 = 0.276812804 * self.s1 - 1.02998914 * self.s2
        print('o2:', (o2 * o2).integrate)

        self.assertAlmostEqual((o1 * o1).integrate, 1, places=4)
        self.assertAlmostEqual((o2 * o2).integrate, 1, places=4)
        self.assertAlmostEqual((o1 * o2).integrate, 0, places=4)

class H2OTest(unittest.TestCase):

    def setUp(self):
        from posym.basis import PrimitiveGaussian, BasisFunction

        mo_coefficients = [
            [0.994216442, 0.025846814, 0.000000000, 0.000000000, -0.004164076, -0.005583712, -0.005583712],
            [0.233766661, -0.844456594, 0.000000000, 0.000000000, 0.122829781, -0.155593214, -0.155593214],
            [0.000000000, 0.000000000, 0.612692349, 0.000000000, 0.000000000, -0.449221684, 0.449221684],
            [-0.104033343, 0.538153649, 0.000000000, 0.000000000, 0.755880259, -0.295107107, -0.295107107],
            [0.000000000, 0.000000000, 0.000000000, -1.000000000, 0.000000000, 0.000000000, 0.000000000],
            [-0.125818566, 0.820120983, 0.000000000, 0.000000000, -0.763538862, -0.769155124, -0.769155124],
            [0.000000000, 0.000000000, 0.959800163, 0.000000000, 0.000000000, 0.814629717, -0.814629717]]


        # Oxigen atom
        sa = PrimitiveGaussian(alpha=130.70932)
        sb = PrimitiveGaussian(alpha=23.808861)
        sc = PrimitiveGaussian(alpha=6.4436083)
        s_O = BasisFunction([sa, sb, sc],
                            [0.154328969, 0.535328136, 0.444634536],
                            center=[0.0000000000, 0.000000000, -0.0808819])  # Bohr

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

        self.molecular_orbitals = []
        for mo in mo_coefficients:
            self.molecular_orbitals.append(np.sum([c*f for c, f in zip(mo, [s_O, s2_O, px_O, py_O, pz_O, s_H, s2_H])]))

    def test_molecular_orbitals_product(self):
        for f1 in self.molecular_orbitals:
            for f2 in self.molecular_orbitals:
                if f1 == f2:
                    self.assertAlmostEqual((f1 * f1).integrate, 1, places=4)
                else:
                    self.assertAlmostEqual((f1 * f2).integrate, 0, places=4)

    def test_mo_symmetry(self):
        mo_sym = [SymmetryFunction('C2v', mo) for mo in self.molecular_orbitals]
        # print(mo_sym)

        a1 = SymmetryBase(group='c2v', rep='A1')
        a2 = SymmetryBase(group='c2v', rep='A2')
        b1 = SymmetryBase(group='c2v', rep='B1')
        b2 = SymmetryBase(group='c2v', rep='B2')

        for mo, ref in zip(mo_sym, [a1, a1, b1, a1, b2, a1, b1]):
            self.assertCountEqual(np.round(mo.get_ir_representation(), decimals=6),
                                  np.round(ref.get_ir_representation(), decimals=6))

    def test_normal_modes(self):
        coordinates = [[ 0.00000, 0.0000000, -0.0808819],
                       [-1.43262, 0.0000000, -1.2823700],
                       [ 1.43262, 0.0000000, -1.2823700]]

        symbols = ['O', 'H', 'H']

        normal_modes = [[[0., 0., -0.075],
                         [-0.381, -0., 0.593],
                         [0.381, -0., 0.593]],  # mode 1

                        [[-0., -0., 0.044],
                         [-0.613, -0., -0.35],
                         [0.613, 0., -0.35]],  # mode 2

                        [[-0.073, -0., -0.],
                         [0.583, 0., 0.397],
                         [0.583, 0., -0.397]]]  # mode 3

        sym_modes_gs = SymmetryModes(group='c2v', coordinates=coordinates, modes=normal_modes, symbols=symbols)

        a1 = SymmetryBase(group='c2v', rep='A1')
        a2 = SymmetryBase(group='c2v', rep='A2')
        b1 = SymmetryBase(group='c2v', rep='B1')
        b2 = SymmetryBase(group='c2v', rep='B2')

        for i, ref in enumerate([a1, a1, b1]):
            nm = sym_modes_gs.get_state_mode(i)
            self.assertCountEqual(np.round(nm.get_ir_representation(), decimals=2),
                                  np.round(ref.get_ir_representation(), decimals=2))

