# SPDX-License-Identifier: GPL-3.0-or-later
# this script checks that everythong is fine, and
# that all the libraries are present


try:
  import entanglement

except:
  print("entanglement library not properly compiled")
  exit()
