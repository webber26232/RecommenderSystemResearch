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
    cdef np.ndarray[np.int32_t, ndim=2] freq
    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.float32_t, ndim=2] prods
    # sum (rxy ^ 2) for common ys
    cdef np.ndarray[np.float32_t, ndim=2] sq
    # sum (rxy) for common ys
    cdef np.ndarray[np.float32_t, ndim=2] s
    # the similarity matrix
    cdef np.ndarray[np.float32_t, ndim=2] sim
    # the std matrix
    cdef np.ndarray[np.float32_t, ndim=2] std

    cdef int i, j, xi, xj, len_y
    cdef np.float32_t ri, rj, self_prod, num, denum
    cdef np.int32_t n, min_sprt
    min_sprt = min_support

    freq = np.zeros((n_x, n_x), np.int32)
    prods = np.zeros((n_x, n_x), np.float32)
    sq = np.zeros((n_x, n_x), np.float32)
    s = np.zeros((n_x, n_x), np.float32)
    sim = np.zeros((n_x, n_x), np.float32)
    std = np.zeros((n_x, n_x), np.float32)

    for y, y_ratings in iteritems(yr):
        len_y = len(y_ratings)
        for i in range(len_y):
            xi, ri = y_ratings[i]
            self_prod = ri * ri
            prods[xi, xi] += self_prod
            freq[xi, xi] += 1
            sq[xi, xi] += self_prod
            s[xi, xi] += ri
            for j in range(i + 1, len_y):
                xj, rj = y_ratings[j]
                prods[xi, xj] += ri * rj
                prods[xj, xi] = prods[xi, xj]
                freq[xi, xj] += 1
                freq[xj, xi] = freq[xi, xj]
                sq[xi, xj] += ri ** 2
                sq[xj, xi] += rj ** 2
                s[xi, xj] += ri
                s[xj, xi] += rj

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            n = freq[xi, xj]
            if n < min_sprt:
                sim[xi, xj] = 0
            else:
                num = n * prods[xi, xj] - s[xi, xj] * s[xj, xi]
                std[xi, xj] = np.sqrt(n * sq[xi, xj] - s[xi, xj] ** 2)
                std[xj, xi] = np.sqrt(n * sq[xj, xi] - s[xj, xi] ** 2)
                denum = std[xi, xj] * std[xj, xi]
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = num / denum
            sim[xj, xi] = sim[xi, xj]

    mean = s / freq
    return mean, std, sim

def mean_std_freq_pearson(n_x, yr):
    """Compute the pairwise overlapping mean & std & sim & freq

    """

    # number of common ys
    cdef np.ndarray[np.int32_t, ndim=2] freq
    # sum (r_xy * r_x'y) for common ys, will be transformed to sim matrix
    cdef np.ndarray[np.float32_t, ndim=2] prods
    # sum (rxy ^ 2) for common ys, will be transformed to std matrix
    cdef np.ndarray[np.float32_t, ndim=2] sq
    # sum (rxy) for common ys, will be transformed to mean matrix
    cdef np.ndarray[np.float32_t, ndim=2] s

    cdef int i, j, xi, xj, len_y
    cdef np.float32_t ri, rj, self_prod, num, denum
    cdef np.int32_t n

    freq = np.zeros((n_x, n_x), np.int32)
    prods = np.zeros((n_x, n_x), np.float32)
    sq = np.zeros((n_x, n_x), np.float32)
    s = np.zeros((n_x, n_x), np.float32)

    for y, y_ratings in iteritems(yr):
        len_y = len(y_ratings)
        for i in range(len_y):
            xi, ri = y_ratings[i]
            self_prod = ri * ri
            prods[xi, xi] += self_prod
            freq[xi, xi] += 1
            sq[xi, xi] += self_prod
            s[xi, xi] += ri
            for j in range(i + 1, len_y):
                xj, rj = y_ratings[j]
                prods[xi, xj] += ri * rj
                prods[xj, xi] = prods[xi, xj]
                freq[xi, xj] += 1
                freq[xj, xi] = freq[xi, xj]
                sq[xi, xj] += ri ** 2
                sq[xj, xi] += rj ** 2
                s[xi, xj] += ri
                s[xj, xi] += rj

    for xi in range(n_x):
        prods[xi, xi] = 1
        n = freq[xi, xi]
        sq[xi, xi] = np.sqrt(n * sq[xi, xi] - s[xi, xi] ** 2) / n
        s[xi, xi] /= n
        for xj in range(xi + 1, n_x):
            n = freq[xi, xj]
            if n > 0:
                sq[xi, xj] = np.sqrt(n * sq[xi, xj] - s[xi, xj] ** 2)
                sq[xj, xi] = np.sqrt(n * sq[xj, xi] - s[xj, xi] ** 2)
                denum = std[xi, xj] * std[xj, xi]
                if denum == 0:
                    prods[xi, xj] = prods[xj, xi] = 0
                else:
                    num = n * prods[xi, xj] - s[xi, xj] * s[xj, xi]
                    prods[xi, xj] = prods[xj, xi] = num / denum

                sq[xi, xj] /= n
                sq[xj, xi] /= n
                s[xi, xj] /= n
                s[xj, xi] /= n

    return s, sq, prods, freq
