# =============================================================================
# Import necessary libraries
# =============================================================================
import pandas as pd
import numpy as np
import os
import ast
from TSC.time_series import generate
from TSC.clustering.TimeSeriesKMeans import TimeSeriesKMeans
from TSC.clustering.TimeSeriesKMedoids import TimeSeriesKMedoids
from TSC.clustering.deep_learning.AEFCNClusterer import AEFCNClusterer
from TSC.clustering.deep_learning.AEResNetClusterer import AEResNetClusterer
from TSC.clustering.feature_based.Catch22Clusterer import Catch22Clusterer
from TSC.clustering.feature_based.TSFreshClusterer import TSFreshClusterer
from TSC.clustering.cross_validation import TimeSeriesCrossValidator
from datetime import datetime
import sys


# =============================================================================
# Set working directory
# =============================================================================
base_working_directory = '/lu/topola/home/mariasad/MS'
os.chdir(base_working_directory)
data_file_path = os.path.join(base_working_directory, 'data_noise_clustering.xlsx')

# =============================================================================
# Load Excel files
# =============================================================================
g_number = int(sys.argv[1])

data = pd.read_excel(data_file_path)
data = data[data['g_number'] == g_number]


# =============================================================================
# Define and create necessary directories
# =============================================================================
g_date = data['g_date'].iloc[0]

# Define working directory based on g_date and g_number
working_directory = os.path.join(base_working_directory, f"{g_date}_{g_number}") 

input_dir = os.path.join(working_directory, 'input_data')
output_dir = os.path.join(working_directory, 'output_data')
log_dir = os.path.join(working_directory, 'logi')
trained_models_dir = os.path.join(working_directory, 'trained_models')

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
        
# Ensure all directories exist
for directory in [input_dir, output_dir, log_dir, trained_models_dir]:
    ensure_directory(directory)

# =============================================================================
# Logging configuration
# =============================================================================
log_filename = f'cross_validation_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'

# =============================================================================
# Generating time_series data if not exists
# =============================================================================
# Get the indexes and exists values for the matched date
random_seeds = eval(data.iloc[0]['d_random_seeds'])   # Extract seeds as a list
g_exists=int(data['g_exists'].iloc[0]) 

if g_exists == 0:
        
    # Now you can extract the required parameters from the current_row
    output_type = str(data['d_output_type'].iloc[0])
    shapes = ast.literal_eval(data['d_shapes'].iloc[0])
    length = int(data['d_length'].iloc[0])
    seed = random_seeds[0]  # You can adjust seed handling as needed
        
    noise_distribution = data['d_noise_distribution'].iloc[0]
    noise_max = data['d_noise_max'].iloc[0]
    parameters_list = eval(data['d_parameters_list'].iloc[0])  # Convert string to list
        
    # Apply noise dynamically if specified
    if pd.notna(noise_distribution) and pd.notna(noise_max):
        noise_max = float(noise_max)
        noise_distribution = str(noise_distribution)

        # Ensure the noise params are set based on the distribution
        if noise_distribution == 'norm':
            noise_params = {'noise': ('norm', 0, noise_max)}
        elif noise_distribution == 'uniform':
            noise_params = {'noise': ('uniform', -noise_max, noise_max)}
        else:
            noise_params = {}

        # Append the noise dictionary as a separate element in the parameters_list
        for i, shape_params in enumerate(parameters_list):
            # Ensure noise is added as a separate dictionary, not inside the shape params
            if isinstance(shape_params[-1], dict) and 'noise' not in shape_params[-1]:
                # If last element is the shape params, append noise as a new dictionary
                shape_params.append(noise_params)
            elif isinstance(shape_params[-1], dict) and 'noise' in shape_params[-1]:
                # Do nothing if noise is already present
                continue
            else:
                # Add noise as a separate dictionary if last element isn't a dict
                shape_params.append(noise_params)

    # Generate the time series data
    time_series, desc = generate.generate_time_series(output_type, shapes, parameters_list, length, seed)

    # Save time_series to a CSV file in ./input_data
    time_series_path = os.path.join(input_dir, 'time_series.csv')
    if isinstance(time_series, pd.DataFrame):
        # If time_series is a DataFrame, save it directly
        time_series.to_csv(time_series_path, index=False)
    elif isinstance(time_series, np.ndarray):
        # If time_series is a numpy array, convert it to DataFrame before saving
        pd.DataFrame(time_series).to_csv(time_series_path, index=False)

    # Save desc to a CSV file in ./input_data
    desc_path = os.path.join(input_dir, 'desc.csv')
    desc.to_csv(desc_path, index=False)
        
# =============================================================================
# Extract cross-validation parameters from the Excel file
# =============================================================================
cv_method = str(data.iloc[0]['d_cv_method'])  # Extract cv_method
n_splits = int(data.iloc[0]['d_n_splits']) if cv_method == 'StratifiedKFold' else None  # Handle n_splits if StratifiedKFold


# =============================================================================
# Helper function to check if a model should be built
# =============================================================================
def check_model_exists(model_col_name):
    return pd.notna(data.iloc[0][model_col_name])  # Return True if data exists in the model column

# =============================================================================
# Podział modeli na procesy
# =============================================================================
# Pobierz nazwy modeli z kolumn Excela
model_columns = [col for col in data.columns if col.startswith('m_')]
models = [col for col in model_columns if check_model_exists(col)]

# =============================================================================
# Helper function for AEFCN 
# =============================================================================
def expand_aefcn_grid(base_grid):
    # wyciągnij listę L (n_layers)
    layers_vals = base_grid.get("n_layers", [3])
    if not isinstance(layers_vals, (list, tuple)):
        layers_vals = [layers_vals]

    grids = []
    for L in layers_vals:
        L = int(L)
        g = dict(base_grid)          # kopia
        g["n_layers"] = [L]          # pojedyncza wartość w siatce

        # Uczyń z list długości L jeden „atom” dla ParameterGrid (lista-lista)
        g["n_filters"]   = [g.get("n_filters",   [64] * L if not isinstance(g.get("n_filters"),   (list, tuple)) else g["n_filters"])]
        g["kernel_size"] = [g.get("kernel_size", [ 3] * L if not isinstance(g.get("kernel_size"), (list, tuple)) else g["kernel_size"])]

        # Jeśli używasz – dopasuj też te pola:
        if "dilation_rate" in g and not (isinstance(g["dilation_rate"], list) and len(g["dilation_rate"]) == L):
            g["dilation_rate"] = [[1] * L]
        if "strides" in g and not (isinstance(g["strides"], list) and len(g["strides"]) == L):
            g["strides"] = [[1] * L]

        # Jeżeli ktoś podał 1-elementową listę (np. [64]) zamiast [64]*L, można też to rozszerzyć:
        def _ensure_len(x, dflt):
            if isinstance(x, list) and len(x) == 1 and L > 1:
                return [x[0]] * L
            return x
        g["n_filters"][0]   = _ensure_len(g["n_filters"][0],   64)
        g["kernel_size"][0] = _ensure_len(g["kernel_size"][0], 3)

        if len(g["n_filters"][0]) != L or len(g["kernel_size"][0]) != L:
            raise ValueError(f"AEFCN: długości n_filters/kernel_size muszą równać się n_layers={L}")

        grids.append(g)
    return grids

def expand_aeresnet_grid(base_grid):
    # wartości R i C mogą być skalarem albo listą -> normalizacja do list
    residual_vals = base_grid.get("n_residual_blocks", [3])
    if not isinstance(residual_vals, (list, tuple)):
        residual_vals = [residual_vals]

    conv_vals = base_grid.get("n_conv_per_residual_block", [3])
    if not isinstance(conv_vals, (list, tuple)):
        conv_vals = [conv_vals]

    def _ensure_per_block_list(val, R, default):
        """
        Zwraca listę długości R:
        - jeśli val to lista o długości R -> użyj
        - jeśli val to lista jednego elementu -> powiel do R
        - jeśli val to skalar -> powiel do R
        """
        if isinstance(val, list):
            if len(val) == R:
                return val
            if len(val) == 1:
                return [val[0]] * R
            # jeśli długość inna niż R i nie 1 -> użyj pierwszej wartości i powiel
            return [val[0]] * R
        return [val] * R

    grids = []
    for R in map(int, residual_vals):
        for C in map(int, conv_vals):
            g = dict(base_grid)  # kopia bazowego grida

            # ustaw pojedyncze wartości dla R i C
            g["n_residual_blocks"] = [R]
            g["n_conv_per_residual_block"] = [C]

            # per-blokowe listy MUSZĄ mieć długość R, opakowane w dodatkową listę (atom)
            nf = _ensure_per_block_list(g.get("n_filters", [64]), R, 64)
            ks = _ensure_per_block_list(g.get("kernel_size", [3]), R, 3)
            g["n_filters"] = [nf]
            g["kernel_size"] = [ks]

            # opcjonalne per-blokowe listy – też do długości R, jeśli są w grida
            for key, dflt in [("dilation_rate", 1), ("strides", 1),
                              ("padding", "same"), ("activation", "relu"),
                              ("use_bias", True)]:
                if key in g:
                    g[key] = [_ensure_per_block_list(g[key], R, dflt)]

            grids.append(g)
    return grids

# =============================================================================
# Dynamically create and evaluate models based on Excel file columns
# =============================================================================
models_to_evaluate = {
    "m_TimeSeriesKMeans": TimeSeriesKMeans,
    "m_TimeSeriesKMedoids": TimeSeriesKMedoids,
    "m_AEFCNClusterer": AEFCNClusterer,
    "m_AEResNetClusterer": AEResNetClusterer,
    "m_Catch22Clusterer": Catch22Clusterer,
    "m_TSFreshClusterer": TSFreshClusterer 
}

for model_name, model_class in models_to_evaluate.items():
    
    # Get the value from the column corresponding to the model
    model_value = data.iloc[0][model_name]

    # Check if the model column exists and has data
    if check_model_exists(model_name):
            # Create the model instance for other models
            model = model_class()
            base_grid = ast.literal_eval(data.iloc[0][model_name])

            if model_name == "m_AEFCNClusterer":
                param_grid = expand_aefcn_grid(base_grid)  # LISTA słowników
            elif model_name=="m_AEResNetClusterer":
                param_grid = expand_aeresnet_grid(base_grid)
            else:
                param_grid = base_grid
                
            # Set up TimeSeriesCrossValidator
            if cv_method == "LeaveOneOut":
                ts_cv = TimeSeriesCrossValidator(
                    model=model,
                    param_grid=param_grid,
                    time_series=time_series,
                    description=desc,
                    cv_method=cv_method,
                    random_seeds=random_seeds,
                    log_dir=log_dir,
                    log_filename=log_filename,
                    dir_output=output_dir
                )
            else:
                ts_cv = TimeSeriesCrossValidator(
                    model=model,
                    param_grid=param_grid,
                    time_series=time_series,
                    description=desc,
                    cv_method="StratifiedKFold",
                    n_splits=n_splits,
                    shuffle=True,
                    random_seeds=random_seeds,
                    log_dir=log_dir,
                    log_filename=log_filename,
                    dir_output=output_dir
                )

            all_params, train_metrics_df, test_metrics_df = ts_cv.cross_validate()


            
            
            
            
            









