#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:33:42 2020

@author: webber
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np

from six.moves import range
from six import iteritems


def mean_std_pearson(n_x, yr, min_support):
    """Compute the pairwise overlapping mean & std & sim 

    """

    # number of common ys
    cdef np.ndarray[np.double_t, ndim=2] freq
    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] prods
    # sum (rxy ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sq
    # sum (rxy) for common ys
    cdef np.ndarray[np.double_t, ndim=2] s
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim
    # the std matrix
    cdef np.ndarray[np.double_t, ndim=2] std

    cdef int i, xi, xj
    cdef double ri, rj, self_prod, n, min_sprt, num, denum
    min_sprt = min_support

    freq = np.zeros((n_x, n_x), np.bouble)
    prods = np.zeros((n_x, n_x), np.double)
    sq = np.zeros((n_x, n_x), np.double)
    s = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)
    std = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for i, (xi, ri) in enumerate(y_ratings):
            self_prod = ri * ri
            prods[xi, xi] += self_prod
            freq[xi, xi] += 1
            sq[xi, xi] += self_prod
            s[xi, xi] += ri
            for xj, rj in y_ratings[i + 1:]:
                prods[xi, xj] += ri * rj
                prods[xj, xi] = prods[xi, xj]
                freq[xi, xj] += 1
                freq[xj, xi] = freq[xj, xi]
                sq[xi, xj] += ri**2
                sq[xj, xi] += rj**2
                s[xi, xj] += ri
                s[xj, xi] += rj

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):

            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                n = freq[xi, xj]
                num = n * prods[xi, xj] - s[xi, xj] * s[xj, xi]
                std[xi, xj] = np.sqrt(n * sq[xi, xj] - s[xi, xj]**2)
                std[xj, xi] = np.sqrt(n * sq[xj, xi] - s[xj, xi]**2)
                denum = std[xi, xj] * std[xj, xi]
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = num / denum

            sim[xj, xi] = sim[xi, xj]

    mean = s / freq
    return mean, std, sim
