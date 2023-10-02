from posym import SymmetryMolecule
import unittest
import numpy as np


class MoleculeTest(unittest.TestCase):

    def test_ir_representation_td(self):
        """
        test the evolution of IR representation with the deformation of a tetrahedron analyzed in Td group

        """

        reference_list = [[ 9.999995e-01 ,  2.775557e-17,  6.394995e-12,  1.522365e-07, 6.888295e-12],
                          [ 9.796177e-01 , -5.551115e-17,  6.569956e-03,  1.134926e-05, 2.402770e-03],
                          [ 9.234299e-01 , -5.551115e-17,  2.415940e-02,  2.624702e-04, 9.154611e-03],
                          [ 8.492140e-01 ,  1.110223e-16,  4.172410e-02,  4.538398e-04, 2.199206e-02],
                          [ 7.736328e-01 ,  1.110223e-16,  5.435149e-02,  7.733429e-04, 3.844804e-02],
                          [ 0.701579     , -0.002867    ,  0.073264    , -0.00179     , 0.053377    ],
                          [ 0.648597     , -0.001133    ,  0.085839    , -0.008104    , 0.06839     ],
                          [ 0.600275     ,  0.00142     ,  0.038918    , -0.001876    , 0.108699    ],
                          [ 0.585275     ,  0.001483    ,  0.031451    , -0.001958    , 0.118738    ]]

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
            np.testing.assert_almost_equal(symmetry_vector, reference_vector, decimal=2)
