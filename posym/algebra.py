# algebra opearation functions for states
import numpy as np


def dot(state1, state2, normalize=False):

    if state1.get_point_group() != state2.get_point_group():
        raise ValueError("States must have same point group")

    pg = state1.get_point_group()
    norm = pg.ir_table.T['E'].values

    v1 = state1.get_ir_representation()
    v2 = state2.get_ir_representation()

    dot = np.sum([np.sqrt((a*b).clip(min=0)) for a, b in zip(np.multiply(v1.values, norm),
                                                             np.multiply(v2.values, norm))])

    if normalize:

        n1 = np.sum(np.multiply(v1.values, norm))
        n2 = np.sum(np.multiply(v2.values, norm))

        dot = dot / np.sqrt(n1 * n2)

    dot = np.square(dot)

    return dot


def norm(state1):

    v1 = state1.get_ir_representation()
    pg = state1.get_point_group()
    norm = pg.ir_table.T['E'].values

    dot = np.sum([a*n for a, n in zip(v1.values, norm)])

    return dot