#!/bin/bash
# A shell script to clean recordings and epoch_checkpoints directory

# Define directories to clean
DIRS=(
    "../epoch_checkpoints"
    "../recordings"
)

# Function to clean a directory
clean_directory() {
    local dir=$1
    if [ -d "$dir" ]; then
        echo "Cleaning directory: $dir"
        rm -rf "$dir"/*
        echo "✓ Contents removed from: $dir"
    else
        echo "⚠️ Directory does not exist: $dir"
    fi
}

# Main script
echo "Starting cleanup..."

# Clean each directory
for dir in "${DIRS[@]}"; do
    clean_directory "$dir"
done

echo "✓ Cleanup complete!"