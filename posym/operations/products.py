from posym.operations.identity import Identity
from posym.operations.reflection import Reflection
from posym.operations.rotation import Rotation
from posym.operations.irotation import ImproperRotation
from posym.operations.inversion import Inversion
from fractions import Fraction
import numpy as np


def test_matrix(matrix1, matrix2):
    if not np.allclose(matrix1, matrix2):
        raise Exception('eps!')


def get_degeneracy(eval, tol=1e-5):

    eval = np.real(eval)

    n_ones = 0
    n_iones = 0
    i_one = None
    i_ione = None

    for i, val in enumerate(eval):
        if 1 - tol < val < 1 + tol:
            n_ones += 1
            i_one = i
        if -1 - tol < val < -1 + tol:
            n_iones += 1
            i_ione = i

    if n_ones == 3:
        return 'E', None
    if n_iones == 3:
        return 'i', None
    if pow(-1, n_iones) == 1:
        return 'r', i_one
    if pow(-1, n_iones) == -1:
        return 's', i_ione

    raise Exception('Operation not recognized')


def get_operation_from_matrix(matrix, tol=1e-5):

    eval, evec = np.linalg.eig(matrix)
    op_type, index = get_degeneracy(eval)

    if op_type == 'E':
        return Identity(label='*E')

    if op_type == 'i':
        return Inversion(label='*i')

    if op_type == 'r':
        axis = np.real(evec.T[index])
        out_axis = [i for i in range(3) if i != index]

        sign = np.sign(np.dot(np.cross(evec.T[out_axis[0]], evec.T[out_axis[1]]), axis).imag)
        sign = sign if sign != 0 else 1

        index_2 = 0 if index != 0 else 1

        # handle numerical error
        eval_2 = np.real(eval[index_2])
        if eval_2 > 1: eval_2 -= 2
        if eval_2 < -1: eval_2 += 2

        angle = np.arccos(eval_2)

        if abs(angle) < tol:
            return Rotation('*R2', axis, order=2, exp=1)

        f = Fraction(2*np.pi/angle).limit_denominator(100)
        order = int(f.numerator)
        exp = int(sign * f.denominator)

        return Rotation('*R{}'.format(order), axis, order=order, exp=exp)

    if op_type == 's':
        axis = np.real(evec.T[index])
        out_axis = [i for i in range(3) if i != index]

        sign = np.sign(np.dot(np.cross(evec.T[out_axis[0]], evec.T[out_axis[1]]), axis).imag)
        sign = sign if sign != 0 else 1

        index_2 = 0 if index != 0 else 1

        if abs(np.real(eval[index_2])-1) < tol:
            test_matrix(Reflection(label='*M', axis=axis).matrix_representation, matrix)
            return Reflection(label='*M', axis=axis)

        # handle numerical error
        eval_2 = np.real(eval[index_2])

        if eval_2 > 1: eval_2 -= 2
        if eval_2 < -1: eval_2 += 2

        angle = np.arccos(eval_2)

        if abs(angle) < tol:
            return Reflection(label='*M', axis=axis)

        f = Fraction(2*np.pi/angle).limit_denominator(5)
        order = int(f.numerator)
        exp = int(sign * f.denominator)

        return ImproperRotation('*S{}'.format(order), axis, order=order, exp=exp)

    raise Exception('Operation not recognized')


def get_operation_from_matrix_new(matrix, tol=1e-5):

    eval, evec = np.linalg.eig(matrix)
    determinat = np.linalg.det(matrix)

    # determine degeneracies and main axis indices
    op_type, index = get_degeneracy(eval)

    if index is None:
        if determinat < 0:
            return Identity(label='*E')
        else:
            return Inversion(label='*I')

    # main axis
    main_axis = np.real(evec.T[index])
    side_axis = [i for i in range(3) if i != index]

    sign = np.sign(np.dot(np.cross(evec.T[side_axis[0]], evec.T[side_axis[1]]), main_axis).imag)
    sign = sign if sign != 0 else 1

    # chose a side axis to get the angle
    index_2 = 0 if index != 0 else 1

    # handle numerical error
    eval_2 = np.real(eval[index_2])
    if eval_2 > 1: eval_2 -= 2
    if eval_2 < -1: eval_2 += 2

    # get angle
    angle = np.arccos(eval_2)

    # rotation route
    if determinat > 0:

        if abs(angle) < tol:
            return Rotation('*R2', main_axis, order=2, exp=1)

        f = Fraction(2*np.pi/angle).limit_denominator(100)
        order = int(f.numerator)
        exp = int(sign * f.denominator)

        return Rotation('*R{}'.format(order), main_axis, order=order, exp=exp)

    # improper rotation route
    if determinat < 0:

        if abs(angle) < tol:
            return Reflection(label='*M', axis=main_axis)

        f = Fraction(2*np.pi/angle).limit_denominator(5)
        order = int(f.numerator)
        exp = int(sign * f.denominator)

        return ImproperRotation('*S{}'.format(order), main_axis, order=order, exp=exp)

    raise Exception('Operation not recognized')


def get_operation_from_matrix_test(matrix, tol=1e-5):

    eval, evec = np.linalg.eig(matrix)
    op_type, index = get_degeneracy(eval)

    if op_type == 'E':
        return Identity(label='*E')

    if op_type == 'i':
        return Inversion(label='*i')

    if op_type == 'r':
        axis = np.real(evec.T[index])
        out_axis = [i for i in range(3) if i != index]

        # print(np.dot(np.cross(evec.T[out_axis[0]], evec.T[out_axis[1]]), axis))
        sign = np.sign(np.dot(np.cross(evec.T[out_axis[0]], evec.T[out_axis[1]]), axis).imag)
        sign = sign if sign != 0 else 1

        index_2 = 0 if index != 0 else 1

        # handle numerical error
        eval_2 = np.real(eval[index_2])
        if eval_2 > 1: eval_2 -= 2
        if eval_2 < -1: eval_2 += 2

        angle = np.arccos(eval_2)

        if abs(angle) < tol:
            return Rotation('*R2', axis, order=2, exp=1)

        f = Fraction(2*np.pi/angle).limit_denominator(100)
        order = int(f.numerator)
        exp = int(sign * f.denominator)

        return Rotation('*R{}'.format(order), axis, order=order, exp=exp)

    if op_type == 's':
        axis = np.real(evec.T[index])
        out_axis = [i for i in range(3) if i != index]

        sign = np.sign(np.dot(np.cross(evec.T[out_axis[0]], evec.T[out_axis[1]]), axis).imag)
        sign = sign if sign != 0 else 1

        index_2 = 0 if index != 0 else 1

        if abs(np.real(eval[index_2])-1) < tol:
            test_matrix(Reflection(label='*M', axis=axis).matrix_representation, matrix)
            return Reflection(label='*M', axis=axis)

        # handle numerical error
        eval_2 = np.real(eval[index_2])

        if eval_2 > 1: eval_2 -= 2
        if eval_2 < -1: eval_2 += 2

        angle = np.arccos(eval_2)

        if abs(angle) < tol:
            return Reflection(label='*M', axis=axis)

        # print('angle: ', np.rad2deg(angle), (2*np.pi)/angle)
        # print('axis: ', axis)

        f = Fraction(2*np.pi/angle).limit_denominator(5)
        # print('Fraction: ', f, f.numerator, f.denominator)
        order = int(f.numerator)
        exp = int(sign * f.denominator)
        # print('order: ', order, 'exp: ', exp)

        return ImproperRotation('*S{}'.format(order), axis, order=order, exp=exp)

    raise Exception('Operation not recognized')
