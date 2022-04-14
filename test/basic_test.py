from posym import PointGroup, SymmetryBase
import unittest
from posym.algebra import dot, norm


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

