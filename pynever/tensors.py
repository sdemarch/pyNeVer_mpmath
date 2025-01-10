"""
This module provides a Tensor class derived from mpmath.matrix

"""

import mpmath
from mpmath import mp

from pynever import MP_PRECISION

mp.dps = MP_PRECISION


class Tensor(mpmath.matrix):
    """Our internal representation of a Tensor of arbitrary precision"""


def ones(shape: tuple[int]) -> mpmath.matrix:
    return mpmath.ones(*shape)


def zeros(shape: tuple[int]) -> mpmath.matrix:
    return mpmath.zeros(*shape)


def reshape(in_tensor: Tensor, new_shape: int | tuple[int]) -> mpmath.matrix:
    raise NotImplementedError

    # def reshape(mat, m, n):  reshape 2d
    #     ...
    #     z = sum(mat.tolist(), [])
    #     ...
    #     return matrix([z[n * k:n * k + n] for k in range(m)])


def array(array_like: list | mpmath.matrix) -> mpmath.matrix:
    return mpmath.matrix(array_like)


def identity(n: int) -> mpmath.matrix:
    return mpmath.eye(n)


def matmul(x1: mpmath.matrix, x2: mpmath.matrix) -> mpmath.matrix:
    return x1 * x2


def vstack(tup: tuple[mpmath.matrix]) -> mpmath.matrix:
    raise NotImplementedError


def hstack(tup: tuple[mpmath.matrix]) -> mpmath.matrix:
    raise NotImplementedError
