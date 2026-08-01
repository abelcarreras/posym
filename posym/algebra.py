# Algebra operations for symmetry state overlap and norm

import numpy as np


def dot(state1, state2, normalize=False):
    """
    Inner product in the G-module space.

    Computes the squared overlap between two symmetry states, weighted
    by irreducible representation degeneracies:

    .. math::

        \\langle c, d \\rangle =
        \\sum_i \\dim(\\Gamma_i) \\cdot c_i \\cdot d_i

    :param state1: first :class:`SymmetryObject`
    :param state2: second :class:`SymmetryObject`
    :param normalize: if True, normalize the result to [0, 1]
    :return: squared inner product as a float
    :raises ValueError: if the two states have different point groups
    """

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
    """
    Norm (total dimension) of a symmetry state.

    Computes the weighted sum of IR coefficients:

    .. math::

        \\|c\\| = \\sum_i \\dim(\\Gamma_i) \\cdot c_i

    :param state1: a :class:`SymmetryObject`
    :return: norm as a float
    """

    v1 = state1.get_ir_representation()
    pg = state1.get_point_group()
    norm = pg.ir_table.T['E'].values

    dot = np.sum([a*n for a, n in zip(v1.values, norm)])

    return dot