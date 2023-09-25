from posym import SymmetryNormalModes, SymmetryAtomDisplacements
from posym import algebra as al
import unittest
import numpy as np
import os
import json


dir_path = os.path.dirname(os.path.realpath(__file__))

class ModesTest(unittest.TestCase):
    longMessage = True


def make_test_function(filename, group):
    def test(self):
        with open(os.path.join(dir_path, 'molecules', filename), 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)

        molecule_coor = json_object['coordinates']
        molecule_symbols = json_object['symbols']
        modes = [m['displacement'] for m in json_object['modes']]
        freqs = [m['frequency'] for m in json_object['modes']]

        sm = SymmetryNormalModes(group=group,
                                 coordinates=molecule_coor,
                                 modes=modes,
                                 symbols=molecule_symbols,
                                 )

        sm_xyz = SymmetryAtomDisplacements(group=group,
                                           coordinates=molecule_coor,
                                           symbols=molecule_symbols)

        def localization(measure_list):
            import itertools
            prod_list = []
            for a in itertools.permutations(measure_list, 2):
                prod_list.append(np.prod(a))
            return np.sum(prod_list)

        pos_measure = sm.measure_pos
        print('fun: ', pos_measure)
        total_loc = []
        for i in range(len(modes)):
            print('m {:2}: {:8.3f} :'.format(i + 1, freqs[i]), sm.get_state_mode(i))
            total_loc.append(localization(sm.get_state_mode(i).get_ir_representation().values))

        max_loc = np.max(np.abs(total_loc))
        norm_diff = np.abs(np.subtract(sm.get_ir_representation().values, sm_xyz.get_ir_representation().values))

        print('Total: ', sm)
        print('Total XYZ: ', sm_xyz)
        print('Norm: ', al.norm(sm), len(molecule_symbols) * 3 - 6)
        print('Dot: ', al.dot(sm, sm))
        print('angles: ', sm.orientation_angles)
        print('loc: ', max_loc)
        print('norm_diff:', np.min(norm_diff))

        self.assertLess(np.min(norm_diff), 1e-2) # check sm == sm_xyz
        self.assertLess(max_loc, 1e-2)
        self.assertLess(pos_measure, 1e-2)

    return test


for j, filename in enumerate(os.listdir(os.path.join(dir_path, 'molecules'))):

    if filename.endswith('json'):
        group = filename.split('_')[0]
        test_func = make_test_function(filename, group)
        setattr(ModesTest, 'test_{0}'.format(filename[:-5]), test_func)
        del test_func
