from posym import SymmetryModes
from posym import algebra as al
import unittest
import numpy as np
import os
import json


class ModesTest(unittest.TestCase):

    def test_from_files(self):
        for j, filename in enumerate(os.listdir()):
            with self.subTest(filename, i=j):

                if filename.endswith('json'):
                    print(filename)
                    group = filename.split('_')[0]

                    with open(filename, 'r') as openfile:
                        # Reading from json file
                        json_object = json.load(openfile)

                    molecule_coor = np.array(json_object['coordinates'])
                    molecule_symbols = np.array(json_object['symbols'])
                    modes = [np.array(m['displacement']) for m in json_object['modes']]
                    freqs = [m['frequency'] for m in json_object['modes']]

                    sm = SymmetryModes(group=group,
                                       coordinates=molecule_coor,
                                       modes=modes,
                                       symbols=molecule_symbols,
                                       )

                    def localization(measure_list):
                        import itertools
                        prod_list = []
                        for a in itertools.permutations(measure_list, 2):
                            prod_list.append(np.prod(a))
                        return np.sum(prod_list)

                    pos_measure = sm.get_measure_pos
                    print('fun: ', pos_measure)
                    total_loc = []
                    for i in range(len(modes)):
                        print('m {:2}: {:8.3f} :'.format(i + 1, freqs[i]), sm.get_state_mode(i))
                        total_loc.append(localization(sm.get_state_mode(i).get_ir_representation().values))

                    max_loc = np.max(np.abs(total_loc))

                    print('Total: ', sm)
                    print('Norm: ', al.norm(sm), len(molecule_symbols) * 3 - 6)
                    print('Dot: ', al.dot(sm, sm))
                    print('angles: ', sm.orientation_angles)
                    print('loc: ', max_loc)

                    self.assertLess(max_loc, 1e-2)
                    self.assertLess(pos_measure, 1e-2)
