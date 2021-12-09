# algebra opearation functions for states
import numpy as np


def dot(state1, state2, normalize=False):

    if state1.get_point_group() != state2.get_point_group():
        raise ValueError("States must have same point group")

    pg = state1.get_point_group()

    v1 = state1.get_ir_representation()
    v2 = state2.get_ir_representation()

    n1 = np.sum(np.multiply(v1.values, pg.ir_degeneracies))
    n2 = np.sum(np.multiply(v2.values, pg.ir_degeneracies))

    dot = np.dot(np.sqrt(np.multiply(v1.values, pg.ir_degeneracies)),
                 np.sqrt(np.multiply(v2.values, pg.ir_degeneracies)))

    if normalize:
        dot = dot / np.sqrt(n1 * n2)

    return dot