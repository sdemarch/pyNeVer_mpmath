"""
This module provides a Tensor class derived from mpmath.matrix

"""

import mpmath
from mpmath import mp

from pynever import MP_PRECISION

mp.dps = MP_PRECISION


class Tensor(mpmath.matrix):
    """Our internal representation of a Tensor of arbitrary precision"""


def ones(shape: tuple[int, ...]) -> mpmath.matrix:
    if len(shape) > 1:
        return mpmath.ones(*shape)
    else:
        return mpmath.ones(*(shape[0], 1))


def zeros(shape: tuple[int, ...]) -> mpmath.matrix:
    if len(shape) > 1:
        return mpmath.zeros(*shape)
    else:
        return mpmath.zeros(*(shape[0], 1))


def reshape_2d(in_tensor: mpmath.matrix, new_shape: tuple[int, ...]) -> mpmath.matrix:
    """
    Procedure to reshape a 2d matrix
    (found on https://groups.google.com/g/mpmath/c/hvVeyEtZOMg)

    """

    z = sum(in_tensor.tolist(), [])

    rows = new_shape[0]
    cols = new_shape[1]

    return mpmath.matrix([z[cols * k:cols * k + cols] for k in range(rows)])


def array(array_like: list | mpmath.matrix) -> mpmath.matrix:
    return mpmath.matrix(array_like)


def identity(n: int) -> mpmath.matrix:
    return mpmath.eye(n)


def matmul(x1: mpmath.matrix, x2: mpmath.matrix) -> mpmath.matrix:
    return x1 * x2


def to_matrix(tup: tuple[mpmath.matrix | mpmath.mpf, ...]) -> tuple[mpmath.matrix, ...]:
    """
    Sanitizer for mixed tuples

    """

    sanitized = ()

    for t in tup:
        if isinstance(t, mpmath.mpf):
            sanitized += (mpmath.matrix([t]),)
        else:
            sanitized += (t,)

    return sanitized


def vstack_2d(tup: tuple[mpmath.matrix | mpmath.mpf, ...]) -> mpmath.matrix:
    """
    Procedure to stack two tensors vertically

    """

    tup = to_matrix(tup)
    cols = tup[0].cols

    # Check compatibility
    for t in tup:
        if t.cols != cols:
            raise Exception('Incorrect dimensions to stack')

    # Create copy of first element
    new_tensor = tup[0].copy()

    # Add dimensions and data
    for t in tup[1:]:
        j = new_tensor.rows
        new_tensor.rows = j + t.rows

        for i in range(t.rows):
            new_tensor[j + i, :] = t[i, :]

    return new_tensor


def hstack_2d(tup: tuple[mpmath.matrix | mpmath.mpf, ...]) -> mpmath.matrix:
    """
    Procedure to stack two tensors horizontally

    """

    tup = to_matrix(tup)
    rows = tup[0].rows

    # Check compatibility
    for t in tup:
        if t.rows != rows:
            raise Exception('Incorrect dimensions to stack')

    # Create copy of first element
    new_tensor = tup[0].copy()

    # Add dimensions and data
    for t in tup[1:]:
        j = new_tensor.cols
        new_tensor.cols = j + t.cols

        for i in range(t.cols):
            new_tensor[:, j + i] = t[:, i]

    return new_tensor
