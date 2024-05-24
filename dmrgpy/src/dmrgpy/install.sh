#!/bin/bash
        
# SPDX-License-Identifier: GPL-3.0-or-later

# Compile ITensor
cd itensor/ITensor-master
make
cd -


# Compile DMRG program
cd itensor
make
cd -



