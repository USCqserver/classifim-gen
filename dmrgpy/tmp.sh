#!/bin/bash

# Function to add license to a file
add_license() {
    file=$1
    license="# SPDX-License-Identifier: GPL-3.0-or-later"

    # Skip files that already have a license
    if grep -q "$license" "$file"; then
        echo "License already exists in $file"
        return
    fi

    # For C++ files, use "//" for comments
    if [[ $file == *".cc" ]] || [[ $file == *".h" ]]; then
        license="// SPDX-License-Identifier: GPL-3.0-or-later"
    fi

    # Check if the file starts with shebang
    if head -1 "$file" | grep -q "^#!"; then
        # Add an empty line after the shebang, then the license
        sed -i "1 a \\
        \n$license" "$file"
    else
        # Add the license at the beginning of the file
        sed -i "1s@^@$license\n@" "$file"
    fi
}

# Read each line from the file
while IFS= read -r line; do
    if [ -f "$line" ]; then
        echo "Adding license to $line"
        add_license "$line"
    else
        echo "File not found: $line"
    fi
done < "./tmp.txt"
