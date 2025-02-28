import os
import subprocess
import re

def find_input_files(root_dir, experiments, time_res, table_id, member, models, variables):
    """
    Find input file paths for CMIP6 models based on the directory structure.

    Args:
    - root_dir (str): Root directory containing CMIP6 model data.
    - experiments (list): List of experiment names.
    - time_res (str): Time resolution directory (e.g., monthly, daily).
    - table_id (str): Table ID to identify model output, e.g. Amon
    - models (list): List of model names.
    - member (str): member example: 'r1i1p1f1'
    - variables (list): List of variable names.

    Returns:
    - input_files (dict): Nested dictionary containing input file paths for each combination of experiment, model, and variable.
    """
    input_files = {}

    for exp in experiments:
        for model in models:
            for var_name in variables:
                directory = os.path.join(root_dir, exp, model, time_res, "global")
                files = [file for file in os.listdir(directory) if file.startswith(f'{var_name}_') and table_id in file and member in file and file.endswith(".nc")]
                if files:
                    sorted_files = sorted([os.path.join(directory, f) for f in files])
                    input_files.setdefault(exp, {}).setdefault(model, {})[var_name] = sorted_files
                
    return input_files


def filter_files_between_years(files, start_year, end_year):
    # Generate pattern for matching years
    pattern = re.compile(r'^.*_(\d{4})(\d{2})-(\d{4})(\d{2})\.nc$')

    # Filter files based on year range
    filtered_files = []
    for file in files:
        match = pattern.match(file)
        if match:
            start_file_year = int(match.group(1))
            end_file_year = int(match.group(3))
            if start_file_year <= end_year and end_file_year >= start_year:
                filtered_files.append(file)

    return filtered_files

def interpolate_to_common_grid(input_files, output_dir, grid, start_year=None, end_year=None, custom_string="", overwrite=False):
    """
    Interpolate CMIP6 models to a common grid using CDO.

    Args:
    - input_files (dict): Nested dictionary containing input file paths for each combination of experiment, model, and variable.
    - output_dir (str): Directory where the interpolated files will be saved.
    - grid (str): A netcdf file containing the grid definition or a string that defines the grid, e.g., r144x72 (2.5deg), r360x180 (1deg).
    - start_year (int, optional): Start year for the time period selection.
    - end_year (int, optional): End year for the time period selection.
    - custom_string (str, optional): Custom string to be added to the output filename.
    - overwrite (bool, optional): If True, overwrite existing files. Default is False.

    Returns:
    - interpolated_files (dict): Nested dictionary containing output file paths for each combination of experiment, model, and variable after interpolation.
    """
    interpolated_files = {}

    for exp, models in input_files.items():
        for model, variables in models.items():
            for var_name, input_files_list in variables.items():
                input_files_list = filter_files_between_years(input_files_list, start_year, end_year)
                if len(input_files_list) == 0:
                    continue

                var_dir = os.path.join(output_dir, var_name)
                os.makedirs(var_dir, exist_ok=True)  # Ensure directory exists

                output_file = os.path.join(var_dir, f"{var_name}_{exp}_{model}_{custom_string}_interpolated_{start_year}-{end_year}.nc")

                # Handle overwriting logic
                if os.path.exists(output_file):
                    if overwrite:
                        os.remove(output_file)  # Remove existing file before proceeding
                        print(f"Overwriting existing file: {output_file}")
                    else:
                        print(f"Output file already exists: {output_file}. Skipping interpolation.")
                        interpolated_files.setdefault(exp, {}).setdefault(model, {})[var_name] = output_file
                        continue

                merged_file = os.path.join(var_dir, f"{var_name}_{exp}_{model}_merged_{start_year}-{end_year}.nc")
                
                # Merge files
                merge_command = ["cdo", "-L", "mergetime", f"-selyear,{start_year}/{end_year}" ] + input_files_list + [merged_file]
                try:
                    subprocess.run(merge_command, check=True)
                    print(f"Merged {var_name} from {model} ({exp}) successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Error merging {var_name} from {model} ({exp}): {e}")
                    continue

                # Interpolate
                remap_command = ["cdo", "-remapbil," + grid, merged_file, output_file]
                try:
                    subprocess.run(remap_command, check=True)
                    print(f"Interpolated {var_name} from {model} ({exp}) successfully.")
                    interpolated_files.setdefault(exp, {}).setdefault(model, {})[var_name] = output_file
                except subprocess.CalledProcessError as e:
                    print(f"Error interpolating {var_name} from {model} ({exp}): {e}")
                    continue

                # Clean up
                try:
                    os.remove(merged_file)
                    print(f"Removed intermediate file: {merged_file}")
                except OSError as e:
                    print(f"Error removing intermediate files: {e}")

    return interpolated_files

def compute_monthly_climatology_cdo(interpolated_files, output_dir, overwrite=False):
    """
    Compute the monthly climatology from the interpolated files for each model and variable using CDO.

    Args:
    - interpolated_files (dict): Nested dictionary containing output file paths for each combination of experiment, model, and variable after interpolation.
    - output_dir (str): Directory where the computed climatology files will be saved.
    - overwrite (bool, optional): If True, overwrite existing climatology files.

    Returns:
    - monthly_climatology_files (dict): Nested dictionary containing file paths of computed monthly climatology for each combination of model and variable.
    """
    monthly_climatology_files = {}

    for exp, models in interpolated_files.items():
        for model, variables in models.items():
            for var_name, file_path in variables.items():
                var_dir = os.path.join(output_dir, var_name)
                os.makedirs(var_dir, exist_ok=True)

                output_file = os.path.join(var_dir, os.path.basename(file_path).replace(".nc", "_climatology.nc"))

                # Handle overwriting logic
                if os.path.exists(output_file):
                    if overwrite:
                        os.remove(output_file)  # Remove existing file before proceeding
                        print(f"Overwriting existing file: {output_file}")
                    else:
                        print(f"Output file already exists: {output_file}. Skipping climatology computation.")
                        monthly_climatology_files.setdefault(exp, {}).setdefault(model, {})[var_name] = output_file
                        continue

                # Compute climatology
                climatology_command = ["cdo", "ymonmean", file_path, output_file]
                try:
                    subprocess.run(climatology_command, check=True)
                    print(f"Computed monthly climatology for {var_name} from {model} ({exp}) successfully.")
                    monthly_climatology_files.setdefault(exp, {}).setdefault(model, {})[var_name] = output_file
                except subprocess.CalledProcessError as e:
                    print(f"Error computing monthly climatology for {var_name} from {model} ({exp}): {e}")

    return monthly_climatology_files


# Example usage
root_dir = "/pool/datos/modelos/cmip6/"
experiments = ["historical"]  # List of experiment names
member='r1i1p1f1'
table_id='Amon'
#table_id='SImon'
time_res = "mm"
grid='r144x72'
#models = [
#    "access-cm2", "access-esm1-5", "awi-cm-1-1-mr", "awi-esm-1-1-lr", "bcc-csm2-mr",
#    "bcc-esm1", "cams-csm1-0", "cas-esm2-0", "cesm2", "cesm2-fv2",
#    "cesm2-waccm", "cesm2-waccm-fv2", "ciesm", "cmcc-cm2-hr4", "cmcc-cm2-sr5",
#    "cmcc-esm2", "canesm5", "canesm5-1", "e3sm-1-0", "e3sm-1-1",
#    "e3sm-1-1-eca", "e3sm-2-0", "e3sm-2-0-narrm", "ec-earth3", "ec-earth3-aerchem",
#    "ec-earth3-cc", "ec-earth3-veg", "ec-earth3-veg-lr", "fgoals-f3-l", "fgoals-g3",
#    "fio-esm-2-0", "giss-e2-1-g", "giss-e2-1-h", "giss-e2-2-g", "giss-e2-2-h",
#    "icon-esm-lr", "iitm-esm", "inm-cm4-8", "inm-cm5-0", "ipsl-cm6a-lr",
#    "ipsl-cm6a-lr-inca", "kace-1-0-g", "kiost-esm", "mcm-ua-1-0", "miroc6",
#    "mpi-esm-1-2-ham", "mpi-esm1-2-hr", "mpi-esm1-2-lr", "mri-esm2-0", "nesm3",
#    "noresm2-lm", "noresm2-mm", "sam0-unicon", "taiesm1"
#]

models = ['awi-cm-1-1-mr', 'bcc-csm2-mr', 'bcc-esm1', 'canesm5', 'canesm5-1', 'cas-esm2-0',
          'cmcc-cm2-hr4', 'cmcc-cm2-sr5', 'cmcc-esm2', 'e3sm-1-1', 'ec-earth3', 'ec-earth3-aerchem',
          'ec-earth3-cc', 'ec-earth3-veg', 'ec-earth3-veg-lr', 'inm-cm5-0', 'mpi-esm1-2-hr', 
          'mpi-esm1-2-lr', 'noresm2-lm']

 # List of model names
variables = ["hur","hus","ta","tas","prw"]  # List of variable names
#variables = ["siconc"] 
output_dir = "/pool/usuarios/bernatj/Data/postprocessed-cmip6/interpolated-2.5deg"

input_files = find_input_files(root_dir, experiments, time_res, table_id, member, models, variables)
interpolated_files = interpolate_to_common_grid(input_files, output_dir, grid, 1980, 2014, member, overwrite=True)

if not os.path.exists(output_dir+'-clim'):
    os.makedirs(output_dir+'-clim')
climatology_files = compute_monthly_climatology_cdo(interpolated_files,output_dir+'-clim', overwrite=True)

interpolated_files = interpolate_to_common_grid(input_files, output_dir, grid, 1850, 1900, member, overwrite=True)
print("Interpolated files:")
print(interpolated_files)

if not os.path.exists(output_dir+'-clim'):
    os.makedirs(output_dir+'-clim')
climatology_files = compute_monthly_climatology_cdo(interpolated_files,output_dir+'-clim', overwrite=True)

