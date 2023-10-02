from posym import SymmetryGaussianLinear, SymmetryObject, SymmetrySingleDeterminant, SymmetryMultiDeterminant
from posym.basis import PrimitiveGaussian, BasisFunction
import posym.algebra as al
import unittest
import numpy as np


class OperationsTest(unittest.TestCase):
    """
    test the symmetry of function based objects (orbitals, density, wave function) in H2O molecule
    """

    def setUp(self):
        #  Oxygen atom
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

        self._basis_set = [s_O, s2_O, px_O, py_O, pz_O, s_H, s2_H]

        # Operate with basis functions in analytic form

        px_O2 = px_O * px_O
        print('integral from -inf to inf:', px_O2.integrate)

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

        self._orbitals = [o1, o2, o3, o4, o5, o6]

    def test_orthogonality(self):
        o1, o2, o3, o4, o5, o6 = self._orbitals

        dot_matrix = np.zeros((5, 5))
        for i, oi in enumerate([o1, o2, o3, o4, o5]):
            for j, oj in enumerate([o1, o2, o3, o4, o5]):
                dot_matrix[i, j] = (oi*oj).integrate

        np.testing.assert_allclose(np.identity(5), dot_matrix, atol=1e-5, rtol=0)

        # Check invariability respect to rotations and displacemnts
        for o in [o1, o2, o3, o4, o5]:
            random_displacement = np.random.random(3)
            random_axis = np.random.random(3)

            o.apply_translation(random_displacement)
            o.apply_rotation(np.pi, random_axis)

            self.assertAlmostEqual((o*o).integrate, 1, places=5)

    def test_orbitals_wf(self):

        o1, o2, o3, o4, o5, o6 = self._orbitals

        a1 = SymmetryObject(group='c2v', rep='A1')
        b1 = SymmetryObject(group='c2v', rep='B1')
        b2 = SymmetryObject(group='c2v', rep='B2')

        sym_o1 = SymmetryGaussianLinear('c2v', o1)
        sym_o2 = SymmetryGaussianLinear('c2v', o2)
        sym_o3 = SymmetryGaussianLinear('c2v', o3)
        sym_o4 = SymmetryGaussianLinear('c2v', o4)
        sym_o5 = SymmetryGaussianLinear('c2v', o5)
        sym_o6 = SymmetryGaussianLinear('c2v', o6)

        print('Symmetry O1: ', sym_o1)
        print('Symmetry O2: ', sym_o2)
        print('Symmetry O3: ', sym_o3)
        print('Symmetry O4: ', sym_o4)
        print('Symmetry O5: ', sym_o5)
        print('Symmetry O6: ', sym_o6)

        self.assertAlmostEqual(al.dot(sym_o1, a1), 1, places=5)
        self.assertAlmostEqual(al.dot(sym_o2, a1), 1, places=5)
        self.assertAlmostEqual(al.dot(sym_o3, b2), 1, places=5)
        self.assertAlmostEqual(al.dot(sym_o4, a1), 1, places=5)
        self.assertAlmostEqual(al.dot(sym_o5, b1), 1, places=5)
        self.assertAlmostEqual(al.dot(sym_o6, a1), 1, places=5)

        # Operate molecular orbitals symmetries to get the symmetry of non-degenerate wave functions

        # restricted close shell
        sym_wf_gs = sym_o1 * sym_o1 * sym_o2 * sym_o2 * sym_o3 * sym_o3 * sym_o4 * sym_o4 * sym_o5 * sym_o5
        print('Symmetry WF (ground state): ', sym_wf_gs)

        # restricted open shell
        sym_wf_excited_1 = sym_o1 * sym_o1 * sym_o2 * sym_o2 * sym_o3 * sym_o3 * sym_o4 * sym_o4 * sym_o5 * sym_o6
        print('Symmetry WF (excited state 1): ', sym_wf_excited_1)

        # restricted close shell
        sym_wf_excited_2 = sym_o1 * sym_o1 * sym_o2 * sym_o2 * sym_o3 * sym_o3 * sym_o4 * sym_o4 * sym_o6 * sym_o6
        print('Symmetry WF (excited state 2): ', sym_wf_excited_2)

        self.assertAlmostEqual(al.dot(sym_wf_gs, a1), 1, places=4)
        self.assertAlmostEqual(al.dot(sym_wf_excited_1, b1), 1, places=4)
        self.assertAlmostEqual(al.dot(sym_wf_excited_2, a1), 1, places=4)

        # apply translation & rotation
        random_displacement = np.random.random(3)
        random_axis = np.random.random(3)
        for o in self._orbitals:
            o.apply_translation(random_displacement)
            o.apply_rotation(np.pi, random_axis)

        print('SymmetryWaveFunction')
        sym_wf_excited_1 = SymmetrySingleDeterminant('c2v',
                                                     [o1, o2, o3, o4, o5],
                                                     [o1, o2, o3, o4, o6])

        print(sym_wf_excited_1.center)
        print('Symmetry WF (excited state 1): ', sym_wf_excited_1)

        sym_wf_excited_2 = SymmetrySingleDeterminant('c2v',
                                                     [o1, o2, o3, o4, o6],
                                                     [o1, o2, o3, o4, o6])
        print('Symmetry WF (excited state 2): ', sym_wf_excited_2)

        self.assertAlmostEqual(al.dot(sym_wf_excited_1, b2) + al.dot(sym_wf_excited_1, b1), 1, places=3)
        self.assertAlmostEqual(al.dot(sym_wf_excited_2, a1), 1, places=3)

        configurations_1 = [{'amplitude': 1.0, 'occupations': {'alpha': [1, 1, 1, 1, 1, 0],
                                                               'beta': [1, 1, 1, 1, 0, 1]}}]

        configurations_2 = [{'amplitude': 1.0, 'occupations': {'alpha': [1, 1, 1, 1, 0, 1],
                                                               'beta': [1, 1, 1, 1, 0, 1]}}]

        print('SymmetryWaveFunctionCI')
        sym_wf_excited_1 = SymmetryMultiDeterminant('c2v', self._orbitals, configurations=configurations_1)
        sym_wf_excited_2 = SymmetryMultiDeterminant('c2v', self._orbitals, configurations=configurations_2)
        print('Symmetry WF (excited state 1): ', sym_wf_excited_1)
        print('Symmetry WF (excited state 2): ', sym_wf_excited_2)
        print(sym_wf_excited_1.center)

        self.assertAlmostEqual(al.dot(sym_wf_excited_1, b2) + al.dot(sym_wf_excited_1, b1), 1, places=3)
        self.assertAlmostEqual(al.dot(sym_wf_excited_2, a1), 1, places=3)
