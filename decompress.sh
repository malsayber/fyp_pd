#!/bin/bash

# Check if a directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Set the directory to start the search from
DIR="$1"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' not found."
    exit 1
fi

# Find all compressed files recursively in the directory
FILES=$(find "$DIR" -type f \( -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.bz2' -o -name '*.tbz2' -o -name '*.tar.xz' -o -name '*.txz' -o -name '*.zip' \))

# Check if any compressed files were found
if [ -z "$FILES" ]; then
    echo "No compressed files found in '$DIR'."
    exit 0
fi

# Loop through the files and extract them
for FILE in $FILES; do
    echo "Extracting $FILE"
    # Create a directory with the same name as the archive (without extension)
    # and extract the contents there.
    TARGET_DIR="${FILE%.*}"
    mkdir -p "$TARGET_DIR"
    tar -xf "$FILE" -C "$TARGET_DIR"
done

echo "Decompression complete."