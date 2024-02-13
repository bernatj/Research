#!/bin/bash

exp=historical

# Set the base directory
base_dir="/pool/usuarios/bernatj/CMIP6/${exp}"

# List of models
models="CESM2-WACCM-FV2 KACE-1-0-G CAMS-CSM1-0 MIROC-ES2L"

# Iterate over each model
for model in $models; do
    # Construct the full directory path
    directory="$base_dir/$model/mm/global"
    
    # Create the directory if it doesn't exist
    mkdir -p "$directory"
    
    # Download data using acccmip6
    acccmip6 -o D -m $model -e $exp -v hur -r atmos -f mon -rlzn 1 -dir "$directory"
done
