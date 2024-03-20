from posym import SymmetryMolecule
import unittest
import numpy as np


class MoleculeTest(unittest.TestCase):

    def test_ir_representation_td(self):
        """
        test the evolution of IR representation with the deformation of a tetrahedron analyzed in Td group

        """

        reference_list = [[ 1.,        0.,        0.,        0.,        0.],
                          [ 0.9796,    0.,        0.0066,    0.,        0.0024],
                          [ 9.24e-01,  0.00e+00,  2.36e-02,  1.00e-04,  9.50e-033],
                          [ 8.492e-01, 0.000e+00, 4.180e-02, 4.000e-04, 2.200e-02],
                          [ 7.729e-01, 0.000e+00, 5.600e-02, 7.000e-04, 3.770e-02],
                          [ 0.7038,    0.,        0.063,     0.0014,    0.0553],
                          [ 0.6454,    0.,        0.0652,    0.0024,    0.0723 ]]

        for x_coor, reference_vector in zip(np.arange(-1.76550, 0, 0.2), reference_list):
            coord = [[x_coor,   0.60223, 0.04154],
                     [-1.68152, 1.00614, 0.61693],
                     [-1.26723, 1.10512, 0.05139],
                     [-1.95144, 1.28518, 0.02485]]

            sm = SymmetryMolecule('Td', coordinates=coord, symbols=['H', 'H', 'H', 'H'])

            # check sm.measure & sm.measure_pos are consistent
            np.testing.assert_almost_equal(sm.measure, sm.measure_pos, decimal=5)

            # check measures with reference
            symmetry_vector = sm.get_ir_representation().values
            print('sym_vect:', np.round(symmetry_vector, decimals=4))
            print('   ref_vect:', np.round(reference_vector, decimals=4))

            np.testing.assert_almost_equal(symmetry_vector, reference_vector, decimal=2)
