# algebra opearation functions for states
import numpy as np


def dot(state1, state2, normalize=False):

    if state1.get_point_group() != state2.get_point_group():
        raise ValueError("States must have same point group")

    pg = state1.get_point_group()

    v1 = state1.get_ir_representation()
    v2 = state2.get_ir_representation()

    n1 = np.sum(v1.values)
    n2 = np.sum(v2.values)

    dot = np.dot(np.sqrt(v1.values.clip(min=0)), np.sqrt(v2.values.clip(min=0)))
    if normalize:
        dot = dot / np.sqrt(n1 * n2)

    return dot