#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:33:42 2020

@author: webber
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport cython
cimport numpy as np  # noqa
import numpy as np

from six.moves import range
from six import iteritems

cdef bint boolean_variable = True

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


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mean_std_freq_pearson(np.int32_t n_x,
                          np.ndarray[np.int32_t, ndim=1] indptr,
                          np.ndarray[np.int32_t, ndim=1] indices,
                          np.ndarray[np.float32_t, ndim=1] data):
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

    cdef int y
    cdef np.float32_t ri, rj, self_prod, num, denum
    cdef np.int32_t i, j, end_y, n, xi, xj

    freq = np.zeros((n_x, n_x), np.int32)
    prods = np.zeros((n_x, n_x), np.float32)
    sq = np.zeros((n_x, n_x), np.float32)
    s = np.zeros((n_x, n_x), np.float32)

    for y in range(indptr.size - 1):
        end_y = indptr[y + 1]
        i = indptr[y]
        while i < end_y:
            xi = indices[i]
            ri = data[i]
            self_prod = ri * ri
            prods[xi, xi] += self_prod
            freq[xi, xi] += 1
            sq[xi, xi] += self_prod
            s[xi, xi] += ri
            j = i + 1
            while j < end_y:
                xj = indices[j]
                rj = data[j]
                prods[xi, xj] += ri * rj
                prods[xj, xi] = prods[xi, xj]
                freq[xi, xj] += 1
                freq[xj, xi] = freq[xi, xj]
                sq[xi, xj] += ri ** 2
                sq[xj, xi] += rj ** 2
                s[xi, xj] += ri
                s[xj, xi] += rj
                j += 1
            i += 1

    xi = 0
    while xi < n_x:
        prods[xi, xi] = 1
        n = freq[xi, xi]
        sq[xi, xi] = np.sqrt(n * sq[xi, xi] - s[xi, xi] ** 2) / n
        s[xi, xi] /= n
        xj = xi + 1
        while xj < n_x:
            n = freq[xi, xj]
            if n > 0:
                sq[xi, xj] = np.sqrt(n * sq[xi, xj] - s[xi, xj] ** 2)
                sq[xj, xi] = np.sqrt(n * sq[xj, xi] - s[xj, xi] ** 2)
                denum = sq[xi, xj] * sq[xj, xi]
                if denum == 0:
                    prods[xi, xj] = prods[xj, xi] = 0
                else:
                    num = n * prods[xi, xj] - s[xi, xj] * s[xj, xi]
                    prods[xi, xj] = prods[xj, xi] = num / denum

                sq[xi, xj] /= n
                sq[xj, xi] /= n
                s[xi, xj] /= n
                s[xj, xi] /= n
            xj += 1
        xi += 1

    return s, sq, prods, freq


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _get_k_neighbors(np.int32_t x, np.int32_t y,
                     np.ndarray[np.int32_t, ndim=1] indptr,
                     np.ndarray[np.int32_t, ndim=1] indices,
                     np.ndarray[np.int32_t, ndim=2] freqs,
                     np.ndarray[np.float32_t, ndim=2] sims,
                     int k, np.int32_t min_support):

    cdef int actual_k, ll, l, rr, r
    cdef np.float32_t sim_p
    cdef np.int32_t start, end, x2, ptr
    cdef np.ndarray[np.int32_t, ndim=1] top_k_pointers

    start = indptr[y]
    end = indptr[y + 1]
    top_k_pointers = np.empty(end - start, dtype=np.int32)

    actual_k = 0
    while start < end:
        x2 = indices[start]
        if (freqs[x, x2] >= min_support) and (sims[x, x2] > 0):
            top_k_pointers[actual_k] = start
            actual_k += 1
        start += 1

    if actual_k > k:
        # perform partition to select top k
        k -= 1
        ll = 0
        rr = actual_k - 1
        while True:
            l = ll
            r = rr
            sim_p = sims[x, indices[top_k_pointers[k]]]

            while l < r:
                if sims[x, indices[top_k_pointers[l]]] > sim_p:
                    l += 1
                else:
                    top_k_pointers[l], top_k_pointers[r] = \
                        top_k_pointers[r], top_k_pointers[l]
                    r -= 1

            if sims[x, indices[top_k_pointers[l]]] <= sim_p:
                l -= 1

            if l > k:
                rr = l
            elif l < k:
                ll = l + 1
            else:
                break

    return top_k_pointers, actual_k


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def intc_Zscore(np.int32_t x, np.int32_t y,
               np.ndarray[np.int32_t, ndim=1] indptr,
               np.ndarray[np.int32_t, ndim=1] indices,
               np.ndarray[np.float32_t, ndim=1] data,
               np.ndarray[np.int32_t, ndim=2] freqs,
               np.ndarray[np.float32_t, ndim=2] sims,
               np.ndarray[np.float32_t, ndim=2] means,
               np.ndarray[np.float32_t, ndim=2] sigmas,
               int k, int min_k, np.int32_t min_support):

    cdef np.ndarray[np.int32_t, ndim=1] top_k_pointers
    cdef np.int32_t x2, ptr
    cdef np.float32_t sum_ratings, sum_sim
    cdef int actual_k, i

    top_k_pointers, actual_k = _get_k_neighbors(
        x, y, indptr, indices, freqs, sims, k, min_support)

    if actual_k < min_k:
        return means[x, x], actual_k

    sum_ratings = 0
    sum_sim = 0
    if actual_k < k:
        k = actual_k

    for i in range(k):
        ptr = top_k_pointers[i]
        x2 = indices[ptr]
        simi = sims[x, x2]
        sum_ratings += simi * ((data[ptr] - means[x2, x])
                               / sigmas[x2, x]
                               * sigmas[x, x2]
                               + means[x, x2])
        sum_sim += simi

    return sum_ratings / sum_sim, actual_k


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def Zscore(np.int32_t x, np.int32_t y,
           np.ndarray[np.int32_t, ndim=1] indptr,
           np.ndarray[np.int32_t, ndim=1] indices,
           np.ndarray[np.float32_t, ndim=1] data,
           np.ndarray[np.int32_t, ndim=2] freqs,
           np.ndarray[np.float32_t, ndim=2] sims,
           np.ndarray[np.float32_t, ndim=2] means,
           np.ndarray[np.float32_t, ndim=2] sigmas,
           int k, int min_k, np.int32_t min_support):

    cdef np.ndarray[np.int32_t, ndim=1] top_k_pointers
    cdef np.int32_t x2, ptr
    cdef np.float32_t sum_ratings, sum_sim, est
    cdef int actual_k, i

    top_k_pointers, actual_k = _get_k_neighbors(
        x, y, indptr, indices, freqs, sims, k, min_support)

    est = means[x, x]

    if actual_k < min_k:
        return est, actual_k

    sum_ratings = 0
    sum_sim = 0
    if actual_k < k:
        k = actual_k

    sum_ratings = 0
    sum_sim = 0
    for i in range(k):
        ptr = top_k_pointers[i]
        x2 = indices[ptr]
        simi = sims[x, x2]
        sum_ratings += simi * ((data[ptr] - means[x2, x2])
                                    / sigmas[x2, x2])

        sum_sim += simi

    est += sum_ratings * sigmas[x, x] / sum_sim
    return est, actual_k


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def with_means(np.int32_t x, np.int32_t y,
               np.ndarray[np.int32_t, ndim=1] indptr,
               np.ndarray[np.int32_t, ndim=1] indices,
               np.ndarray[np.float32_t, ndim=1] data,
               np.ndarray[np.int32_t, ndim=2] freqs,
               np.ndarray[np.float32_t, ndim=2] sims,
               np.ndarray[np.float32_t, ndim=2] means,
               int k, int min_k, np.int32_t min_support):

    cdef np.ndarray[np.int32_t, ndim=1] top_k_pointers
    cdef np.int32_t x2, ptr
    cdef np.float32_t sum_ratings, sum_sim, est
    cdef int actual_k, i

    top_k_pointers, actual_k = _get_k_neighbors(
        x, y, indptr, indices, freqs, sims, k, min_support)

    est = means[x, x]

    if actual_k < min_k:
        return est, actual_k

    sum_ratings = 0
    sum_sim = 0
    if actual_k < k:
        k = actual_k

    sum_ratings = 0
    sum_sim = 0
    for i in range(k):
        ptr = top_k_pointers[i]
        x2 = indices[ptr]
        simi = sims[x, x2]
        sum_ratings += simi * (data[ptr] - means[x2, x2])
        sum_sim += simi

    est += sum_ratings / sum_sim
    return est, actual_k


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def basic(np.int32_t x, np.int32_t y,
          np.ndarray[np.int32_t, ndim=1] indptr,
          np.ndarray[np.int32_t, ndim=1] indices,
          np.ndarray[np.float32_t, ndim=1] data,
          np.ndarray[np.int32_t, ndim=2] freqs,
          np.ndarray[np.float32_t, ndim=2] sims,
          np.ndarray[np.float32_t, ndim=2] means,
          int k, int min_k, np.int32_t min_support):

    cdef np.ndarray[np.int32_t, ndim=1] top_k_pointers
    cdef np.int32_t ptr
    cdef np.float32_t sum_ratings, sum_sim
    cdef int actual_k, i

    top_k_pointers, actual_k = _get_k_neighbors(
        x, y, indptr, indices, freqs, sims, k, min_support)

    if actual_k < min_k:
        return means[x, x], actual_k

    sum_ratings = 0
    sum_sim = 0
    if actual_k < k:
        k = actual_k

    sum_ratings = 0
    sum_sim = 0
    for i in range(k):
        ptr = top_k_pointers[i]
        simi = sims[x, indices[ptr]]
        sum_ratings += simi * data[ptr]
        sum_sim += simi

    return sum_ratings / sum_sim, actual_k


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _fcp(np.ndarray[np.float32_t, ndim=1] rating,
         np.ndarray[np.float32_t, ndim=1] est,
         np.ndarray[np.int64_t, ndim=1] count,
         bint weight):
    cdef int nc, nd, i, j, k, ncu, ndu
    cdef np.int64_t l, r, u_count

    nc = 0
    nd = 0
    for i in range(count.size - 1):
        ncu = 0
        ndu = 0
        l = count[i]
        r = count[i + 1]

        if weight:
            u_count = r - l
        else:
            u_count = 1

        for j in range(l, r):
            for k in range(j + 1, r):
                if ((est[j] > est[k] and rating[j] > rating[k])
                    or (est[j] < est[k] and rating[j] < rating[k])):
                    ncu += 1
                if ((est[j] >= est[k] and rating[j] < rating[k])
                    or (est[j] <= est[k] and rating[j] > rating[k])):
                    ndu += 1
        nc += ncu * u_count
        nd += ndu * u_count
    return nc / (nc + nd)
