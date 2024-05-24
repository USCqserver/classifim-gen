# SPDX-License-Identifier: GPL-3.0-or-later

from numba import jit


def jsum(n):
  a = 0.0
  for i in range(n):
    for j in range(n):
      a += i - j
  return a

print(jsum(10000))


