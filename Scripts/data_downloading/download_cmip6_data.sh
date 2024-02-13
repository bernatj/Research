#!/bin/bash

#before downloading the data you can check which models are availabel by exectuting the following command.
#e.g. for relative humidity (hur) monthly means, one realization
#acccmip6 -o S -e historical -v hur -r atmos -f mon -rlzn 1 

#CMIP6 experiment
exp=historical
# Set the base directory
base_dir="/pool/datos/modelos/cmip6/${exp}" 
# List of models
models="E3SM-2-0-NARRM EC-Earth3-AerChem CNRM-ESM2-1 NorESM2-MM CESM2-WACCM IITM-ESM EC-Earth3 IPSL-CM6A-LR-INCA NESM3 CMCC-CM2-HR4 \
        CAMS-CSM1-0 E3SM-2-0 ACCESS-CM2 CNRM-CM6-1 ACCESS-ESM1-5 BCC-CSM2-MR CESM2-FV2 SAM0-UNICON GISS-E2-2-G TaiESM1 CanESM5 INM-CM5-0 \
        FGOALS-g3 AWI-CM-1-1-MR CESM2 HadGEM3-GC31-MM UKESM1-0-LL FIO-ESM-2-0 MIROC6 MPI-ESM-1-2-HAM GISS-E2-2-H MIROC-ES2H MPI-ESM2-0 \
        GISS-E2-1-G BCC-ESM1 E3SM-1-1"

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
        acccmip6 -o D -m $model -e historical -v $variable -r atmos -f mon -rlzn 1 -dir "$directory"
    done
done
