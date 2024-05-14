#!/bin/bash

var="ta"
# Directory containing the input files
input_dir="/home/bernatj/Data/postprocessed-cmip6/interpolated-2.5deg-clim/${var}"

# Directory to store the output files
output_dir="/home/bernatj/Data/postprocessed-cmip6/interpolated-2.5deg-multimodel/${var}"
# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

echo $input_dir
# Model list (for multimodel mean there shouldn't be missing values)
models=('ec-earth3-veg-lr' 'ec-earth3-cc' 'awi-cm-1-1-mr' 'canesm5-1' 'cmcc-cm2-sr5'
        'cmcc-cm2-hr4' 'ec-earth3-veg'  'bcc-csm2-mr' 'cams-csm1-0' 'cmcc-esm2')

# Array to store input files for multimodel mean
input_files=()

# Iterate through each model
for model in "${models[@]}"; do
    echo $model 

    input_file="$input_dir/${var}_${model}_delta.nc"
    echo $input_dir
    # Check if both files exist
    if [ -f "$input_file" ]; then
        input_files+=($input_file)
    else
        echo "Files not found for $model"
    fi
done

# Check if there are input files to compute multimodel mean
if [ ${#input_files[@]} -gt 0 ]; then
    # Output file name
    output_file="$output_dir/${var}_multimodel_mean.nc"
    
    # Compute multimodel mean using CDO
    cdo ensmean "${input_files[@]}" "$output_file"
else
    echo "No input files found for multimodel mean."
fi
