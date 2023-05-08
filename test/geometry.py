from posym import SymmetryMoleculeBase
import unittest
import numpy as np


class MoleculeTest(unittest.TestCase):

    def test_td(self):

        reference_list = [[ 9.99999543e-01,  2.77555756e-17,  6.39499564e-12,  1.52236556e-07, 6.88829549e-12],
                          [ 9.79617728e-01, -5.55111512e-17,  6.56995637e-03,  1.13492681e-05, 2.40277058e-03],
                          [ 9.23429943e-01, -5.55111512e-17,  2.41594059e-02,  2.62470299e-04, 9.15461149e-03],
                          [ 8.49214078e-01,  1.11022302e-16,  4.17241083e-02,  4.53839880e-04, 2.19920620e-02],
                          [ 7.73632843e-01,  1.11022302e-16,  5.43514975e-02,  7.73342935e-04, 3.84480443e-02],
                          [ 0.69634141    , -0.00195505    ,  0.07542687    ,  0.0012846     , 0.05030203    ],
                          [ 6.47426818e-01,  3.69769797e-05,  8.58392193e-02, -7.93371080e-03, 6.82196328e-02],
                          [ 0.59920578    ,  0.00061109    ,  0.03985695    , -0.00200576    , 0.10882883    ],
                          [ 0.58445688    ,  0.00110784    ,  0.03204778    , -0.00217935    , 0.11895925    ]]

        for x_coor, reference_vector in zip(np.arange(-1.76550, 0, 0.2), reference_list):
            coord = [[x_coor,   0.60223, 0.04154],
                     [-1.68152, 1.00614, 0.61693],
                     [-1.26723, 1.10512, 0.05139],
                     [-1.95144, 1.28518, 0.02485]]

            sm = SymmetryMoleculeBase('Td', coordinates=coord, symbols=['H', 'H', 'H', 'H'])

            # check sm.measure & sm.measure_pos are consistent
            np.testing.assert_almost_equal(sm.measure, sm.measure_pos, decimal=8)

            # check measures with reference
            symmetry_vector = sm.get_ir_representation().values
            np.testing.assert_almost_equal(symmetry_vector, reference_vector, decimal=6)
