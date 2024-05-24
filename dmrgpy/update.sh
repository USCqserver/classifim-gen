#!/bin/bash
        
# SPDX-License-Identifier: GPL-3.0-or-later

python clean.py
git add .
git commit -m "Update"
git push
