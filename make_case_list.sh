#!/bin/bash

# ==========================================
# CREATE GLOBAL CASE LIST
# ==========================================

BASE_DIR="/scratch/work/langercostaw/swash_landes/swash_cases/swash_runs"

OUTPUT_FILE="$BASE_DIR/case_list.txt"

cd "$BASE_DIR" || exit 1

echo "Searching transect folders..."

# Remove old list if exists
rm -f "$OUTPUT_FILE"

# Loop through transect folders
for transect in */ ; do

    echo "Processing transect: $transect"

    find "$transect" -maxdepth 1 -type d -name "case_*" | sort >> "$OUTPUT_FILE"

done

# Remove possible duplicates
sort -u "$OUTPUT_FILE" -o "$OUTPUT_FILE"

# Count cases
NCASES=$(wc -l < "$OUTPUT_FILE")

echo "--------------------------------------"
echo "case_list.txt created"
echo "Total cases: $NCASES"
echo "Location: $OUTPUT_FILE"
echo "--------------------------------------"

# Preview
head "$OUTPUT_FILE"
