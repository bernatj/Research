#!/bin/bash

#before downloading the data you can check which models are availabel by exectuting the following command.
#e.g. for relative humidity (hur) monthly means, one realization
#acccmip6 -o S -e historical -v hur -r atmos -f mon -rlzn 1 

#CMIP6 experiment
exp=historical
# Set the base directory
base_dir="/pool/datos/modelos/cmip6/${exp}" 
# List of models
#models="E3SM-2-0-NARRM EC-Earth3-AerChem CNRM-ESM2-1 NorESM2-MM CESM2-WACCM IITM-ESM EC-Earth3 IPSL-CM6A-LR-INCA NESM3 CMCC-CM2-HR4 \
#        CAMS-CSM1-0 E3SM-2-0 ACCESS-CM2 CNRM-CM6-1 ACCESS-ESM1-5 BCC-CSM2-MR CESM2-FV2 SAM0-UNICON GISS-E2-2-G TaiESM1 CanESM5 INM-CM5-0 \
#        FGOALS-g3 AWI-CM-1-1-MR CESM2 HadGEM3-GC31-MM UKESM1-0-LL FIO-ESM-2-0 MIROC6 MPI-ESM-1-2-HAM GISS-E2-2-H MIROC-ES2H MPI-ESM2-0 \
#        GISS-E2-1-G BCC-ESM1 E3SM-1-1"

models=(
    "ACCESS-CM2" "ACCESS-ESM1-5" "AWI-CM-1-1-MR" "AWI-ESM-1-1-LR" "BCC-CSM2-MR"
    "BCC-ESM1" "CAMS-CSM1-0" "CAS-ESM2-0" "CESM2" "CESM2-FV2"
    "CESM2-WACCM" "CESM2-WACCM-FV2" "CIESM" "CMCC-CM2-HR4" "CMCC-CM2-SR5"
    "CMCC-ESM2" "CanESM5" "CanESM5-1" "E3SM-1-0" "E3SM-1-1"
    "E3SM-1-1-ECA" "E3SM-2-0" "E3SM-2-0-NARRM" "EC-Earth3" "EC-Earth3-AerChem"
    "EC-Earth3-CC" "EC-Earth3-Veg" "EC-Earth3-Veg-LR" "FGOALS-f3-L" "FGOALS-g3"
    "FIO-ESM-2-0" "GISS-E2-1-G" "GISS-E2-1-H" "GISS-E2-2-G" "GISS-E2-2-H"
    "ICON-ESM-LR" "IITM-ESM" "INM-CM4-8" "INM-CM5-0" "IPSL-CM6A-LR"
    "IPSL-CM6A-LR-INCA" "KACE-1-0-G" "KIOST-ESM" "MCM-UA-1-0" "MIROC6"
    "MPI-ESM-1-2-HAM" "MPI-ESM1-2-HR" "MPI-ESM1-2-LR" "MRI-ESM2-0" "NESM3"
    "NorESM2-LM" "NorESM2-MM" "SAM0-UNICON" "TaiESM1"
)

#exclude options (we only one the r1i1p1f1 member)
skip='1p2,1p3,1p4,1p5,2p2,3p2,1f2,CFmon'

# List of variables separated by space (hur va ua ta tas tos hurs)
variables="hur ta ua va"     

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
        acccmip6 -o D -m $model -e historical -v $variable -r atmos -f mon -rlzn 1 -skip $skip -dir "$directory"
    done
done
