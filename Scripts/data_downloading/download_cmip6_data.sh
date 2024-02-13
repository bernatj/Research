#!/bin/bash

exp=historical

# Set the base directory
base_dir="/pool/usuarios/bernatj/CMIP6/${exp}" 
# List of models
models="CESM2-WACCM-FV2 KACE-1-0-G CAMS-CSM1-0 MIROC-ES2L"
# List of variables separated by space
variables="hur"     

# Iterate over each model
for model in $models; do
    # Convert model name to lowercase
    model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
    
    # Construct the full directory path
    directory="$base_dir/$model_lower/mm/global"
    
    # Create the directory if it doesn't exist
    mkdir -p "$directory"
    
    # Iterate over each variable
    for variable in $variables; do
        # Download data using acccmip6
        acccmip6 -o D -m $model -e historical -v $variable -r atmos -f mon -rlzn 1 -d "$directory"
    done
done
