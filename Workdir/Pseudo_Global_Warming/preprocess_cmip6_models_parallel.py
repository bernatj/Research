import os
import subprocess
import re
from multiprocessing import Pool, cpu_count

def find_input_files(root_dir, experiments, time_res, table_id, member, models, variables):
    """Find input file paths for CMIP6 models based on the directory structure."""
    input_files = {}
    for exp in experiments:
        for model in models:
            for var_name in variables:
                directory = os.path.join(root_dir, exp, model, time_res, "global")
                if not os.path.exists(directory):
                    continue  # Skip if directory does not exist

                files = [
                    os.path.join(directory, file)
                    for file in os.listdir(directory)
                    if file.startswith(f"{var_name}_") and table_id in file and member in file and model in file.lower and file.endswith(".nc")
                ]
                if files:
                    input_files.setdefault(exp, {}).setdefault(model, {})[var_name] = sorted(files)
                
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

def process_model_climatology(args):
    """Compute monthly climatology for a given model-variable in parallel."""
    exp, model, var_name, input_files_list, output_dir, start_year, end_year, overwrite = args
    input_files_list = filter_files_between_years(input_files_list, start_year, end_year)
    if not input_files_list:
        return (exp, model, var_name, None)

    var_dir = os.path.join(output_dir, var_name)
    os.makedirs(var_dir, exist_ok=True)

    output_file = os.path.join(var_dir, f"{var_name}_{exp}_{model}_climatology_{start_year}-{end_year}.nc")
    if os.path.exists(output_file) and not overwrite:
        print(f"Skipping climatology, output exists: {output_file}")
        return (exp, model, var_name, output_file)

    merged_file = os.path.join(var_dir, f"{var_name}_{exp}_{model}_merged_{start_year}-{end_year}.nc")
    merge_command = ["cdo", "-L", "mergetime", f"-selyear,{start_year}/{end_year}"] + input_files_list + [merged_file]
    climatology_command = ["cdo", "-O", "ymonmean", merged_file, output_file]

    try:
        subprocess.run(merge_command, check=True)
        subprocess.run(climatology_command, check=True)
        os.remove(merged_file)
        return (exp, model, var_name, output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error computing climatology for {var_name} from {model} ({exp}): {e}")
        return (exp, model, var_name, None)

def process_model_interpolation(args):
    """Interpolate climatology files to a common grid in parallel."""
    exp, model, var_name, file_path, output_dir, grid, overwrite = args
    if not file_path:
        return (exp, model, var_name, None)

    var_dir = os.path.join(output_dir, var_name)
    os.makedirs(var_dir, exist_ok=True)

    output_file = os.path.join(var_dir, os.path.basename(file_path).replace(".nc", f"_{grid}_interpolated.nc"))
    if os.path.exists(output_file) and not overwrite:
        print(f"Skipping interpolation, output exists: {output_file}")
        return (exp, model, var_name, output_file)

    remap_command = ["cdo", f"-remapbil,{grid}", file_path, output_file]
    try:
        subprocess.run(remap_command, check=True)
        return (exp, model, var_name, output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error interpolating {var_name} from {model} ({exp}): {e}")
        return (exp, model, var_name, None)

def parallel_climatology(input_files, output_dir, start_year, end_year, overwrite):
    """Run climatology computation in parallel with dynamic scheduling."""
    args_list = [
        (exp, model, var_name, input_files[exp][model][var_name], output_dir, start_year, end_year, overwrite)
        for exp in input_files
        for model in input_files[exp]
        for var_name in input_files[exp][model]
    ]

    climatology_files = {}

    with Pool(processes=cpu_count()) as pool:
        for exp, model, var_name, file_path in pool.imap_unordered(process_model_climatology, args_list):
            if file_path:
                climatology_files.setdefault(exp, {}).setdefault(model, {})[var_name] = file_path

    return climatology_files

def parallel_interpolation(climatology_files, output_dir, grid, overwrite):
    """Run interpolation in parallel with dynamic scheduling."""
    args_list = [
        (exp, model, var_name, climatology_files[exp][model][var_name], output_dir, grid, overwrite)
        for exp in climatology_files
        for model in climatology_files[exp]
        for var_name in climatology_files[exp][model]
    ]

    interpolated_files = {}

    with Pool(processes=cpu_count()) as pool:
        for exp, model, var_name, file_path in pool.imap_unordered(process_model_interpolation, args_list):
            if file_path:
                interpolated_files.setdefault(exp, {}).setdefault(model, {})[var_name] = file_path

    return interpolated_files

# Example usage
root_dir = "/pool/datos/modelos/cmip6/"
experiments = ["historical"]
member = 'r1i1p1f1'
table_id = 'Amon'
time_res = "mm"
grid = 'r144x72'

models = ['awi-cm-1-1-mr', 'bcc-csm2-mr', 'bcc-esm1', 'cams-csm1-0', 'canesm5', 'canesm5-1', 'cas-esm2-0',\
          'cmcc-cm2-hr4', 'cmcc-cm2-sr5', 'cmcc-esm2', 'e3sm-1-1', 'ec-earth3', 'ec-earth3-aerchem', \
          'ec-earth3-cc', 'ec-earth3-veg', 'ec-earth3-veg-lr']

variables = ["hur", "hus", "ta", "tas", "prw"]
output_dir = "/pool/usuarios/bernatj/Data/postprocessed-cmip6/climatology"

input_files = find_input_files(root_dir, experiments, time_res, table_id, member, models, variables)

# Compute Climatology First (1980-2014)
climatology_files = parallel_climatology(input_files, output_dir, 1980, 2014, overwrite=False)

# Interpolate Climatology
interpolated_output_dir = output_dir + '-interpolated-2p5deg'
os.makedirs(interpolated_output_dir, exist_ok=True)
interpolated_files = parallel_interpolation(climatology_files, interpolated_output_dir, grid, overwrite=False)

# Repeat for another time period (1850-1900)
climatology_files = parallel_climatology(input_files, output_dir, 1850, 1900, overwrite=False)
interpolated_files = parallel_interpolation(climatology_files, interpolated_output_dir, grid, overwrite=False)

print("Final Interpolated Climatology Files:", interpolated_files)
