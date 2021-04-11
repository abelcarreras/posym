import numpy as np
from itertools import permutations


class Operation:
    def __init__(self, coordinates, symbols):
        self._coordinates = coordinates
        self._symbols = np.zeros(len(coordinates)) if symbols is None else np.array(symbols)

        self._measure_mode = []
        self._measure_coor = []

    def get_permutation(self, operation):
        operated_coor = np.dot(operation, self._coordinates.T).T

        coor_list = []
        permu_list = []
        for iter in permutations(enumerate(operated_coor), len(operated_coor)):

            iter_num = [c[0] for c in iter]
            if not (self._symbols[iter_num] == self._symbols).all():
                continue

            permu_list.append(iter_num)
            permu_coor = np.array([c[1] for c in iter])

            coor_list.append(np.average(np.linalg.norm(np.subtract(self._coordinates, permu_coor), axis=0)))

        return np.min(coor_list), permu_list[np.nanargmin(coor_list)]

    def get_measure(self):
        return np.array(self._measure_mode)

    def get_coor_measure(self):
        #  normalization

        sum_list = []
        for r1 in self._coordinates:
            for r2 in self._coordinates:
                subs = np.subtract(r1, r2)
                sum_list.append(np.dot(subs, subs))
        d = np.average(sum_list)

        return np.average(self._measure_coor) / d


if __name__ == '__main__':

    from pyqchem import get_output_from_qchem, Structure, QchemInput
    from pyqchem.parsers.parser_frequencies import basic_frequencies
    from posym.operations.rotation import Rotation
    from posym.operations.reflection import Reflection


    m_measure = []
    for x_coor in np.arange(-1.0, 1.0, 0.02):
        water = [[ x_coor,  0.00000000e+00,  2.40297090e-01],
                 [-1.43261539e+00, -1.75444785e-16, -9.61188362e-01],
                 [ 1.43261539e+00,  1.75444785e-16, -9.61188362e-01]]

        symbols = ['O', 'H', 'H']
        molecule = Structure(coordinates=np.array(water) * 0.5,
                             symbols=symbols,
                             charge=0,
                             multiplicity=1)

        qc_input = QchemInput(molecule,
                              jobtype='freq',
                              exchange='hf',
                              basis='6-31G',
                              # sym_ignore=True,
                              )

        parsed_data, ee = get_output_from_qchem(qc_input, parser=basic_frequencies, read_fchk=True)

        water = np.array(ee['structure'].get_coordinates())

        #print(' structure')
        #print('Final energy:', parsed_data['scf_energy'])

        modes = [np.array(m['displacement']) for m in parsed_data['modes']]
        freqs = [m['frequency'] for m in parsed_data['modes']]

        c2 = Rotation(water, modes, [0, 0, 1], order=2, symbols=symbols)
        r_yz = Reflection(water, modes, [1, 0, 0], symbols=symbols)
        r_xz = Reflection(water, modes, [0, 1, 0], symbols=symbols)
        if False:

            print('Rotation (c2)')
            print('measure mode: ', c2.get_measure())
            print('measure coor', c2.get_coor_measure())

            print('Reflection (yz)')
            print('measure mode: ', r_yz.get_measure())
            print('measure coor', r_yz.get_coor_measure())

            print('Reflection (xz)')
            print('measure mode: ', r_xz.get_measure())
            print('measure coor', r_xz.get_coor_measure())

        from posym import PointGroup, SymmetryBase, SymmetryModes
        import pandas as pd

        pg = PointGroup(group='C2v')

        m = 2  # mode number
        state = SymmetryBase(group='C2v',
                             rep=pd.Series([1, c2.get_measure()[m], r_yz.get_measure()[m], r_xz.get_measure()[m]],
                                           index=['E', 'C2', 'sv_xz', 'sv_yz']))
        #print('state: ', state.get_ir_representation().values)
        #m_measure.append(state.get_ir_representation().values)

        sm = SymmetryModes(group='C2v', coordinates=water, modes=modes, symbols=symbols)

        m_measure.append(sm.get_state_mode(1).get_ir_representation().values)


    import matplotlib.pyplot as plt
    for m, l  in zip(np.array(m_measure).T, state.get_ir_representation().index):
        plt.plot(np.arange(-1.0, 1.0, 0.02), m, '-', label=l)

    plt.plot(np.arange(-1.0, 1.0, 0.02), np.sum(m_measure, axis=1), '-', label='sum')

    plt.legend()
    plt.show()
