import os
import numpy as np
import matplotlib.pyplot as plt
from ioh import get_problem, ProblemClass
import ioh.iohcpp.logger as logger_lib
from ioh.iohcpp.logger import Analyzer
from ioh.iohcpp.logger.trigger import Each, ON_IMPROVEMENT, ALWAYS
from Algorithms import Vanilla_BO
import pandas as pd
from pathlib import Path

# Configuration
problem_id = 18  
dimension = 2   # 2D function
instances = range(5)  # Instances 0 to 4
budget = 100  # Function evaluations per run
n_DoE = 10    # Initial design of experiments size
acquisition_function = "upper_confidence_bound"
random_seed = 42

# Directory to save results
results_dir = "convergence_results"
os.makedirs(results_dir, exist_ok=True)

# Function to extract convergence data from log files
def extract_convergence_data(instance):
    # The folder name will be created by the logger
    folder_name = f"my-experiment-{instance}"
    
    print(f"Looking for data in folder: {folder_name}")
    
    # Find the most recent folder if there are multiple with the same name
    base_path = Path(os.getcwd()) / folder_name
    if not base_path.exists():
        print(f"Warning: Folder {folder_name} not found")
        return None
    
    # Look for the data folder (should be named data_f{problem_id}_Sphere)
    data_folders = list(base_path.glob(f"data_f{problem_id}_*"))
    if not data_folders:
        print(f"Warning: No data folder found in {folder_name}")
        return None
    
    data_folder = data_folders[0]
    print(f"Found data folder: {data_folder}")
    
    # Find the data file in the data folder
    data_files = list(data_folder.glob("*.dat"))
    if not data_files:
        print(f"Warning: No data file found in {data_folder}")
        return None
    
    data_file = data_files[0]
    print(f"Found data file: {data_file}")
    
    # Read the data file
    try:
        data = pd.read_csv(data_file, sep="\s+")  # Use whitespace as separator
        print(f"Columns in data file: {data.columns.tolist()}")
        
        # Extract function evaluations and best-so-far values
        evals = data["evaluations"].values
        best_so_far = data["raw_y_best"].values  # Use raw_y_best instead of raw_y
        
        return evals, best_so_far
    except Exception as e:
        print(f"Error reading data file: {e}")
        return None

# Run optimization for each instance
all_results = []

for instance in instances:
    print(f"\nRunning optimization for instance {instance}...")
    
    # Set up logger for this instance
    triggers = [
        Each(1),  # Log after every evaluation
        ON_IMPROVEMENT  # Log when there's an improvement
    ]
    
    logger = Analyzer(
        triggers=triggers,
        root=os.getcwd(),
        folder_name=f"my-experiment-{instance}",
        algorithm_name=f"Vanilla BO Instance {instance}",
        algorithm_info="Bo-Torch Implementation",
        additional_properties=[logger_lib.property.RAWYBEST],
        store_positions=True
    )
    
    # Create the problem with the current instance
    problem = get_problem(problem_id, 
                         instance=instance, 
                         dimension=dimension,
                         problem_class=ProblemClass.BBOB)
    
    # Attach logger to the problem
    problem.attach_logger(logger)
    
    # Set up the Vanilla BO optimizer
    optimizer = Vanilla_BO(budget=budget,
                          n_DoE=n_DoE,
                          acquisition_function=acquisition_function,
                          random_seed=random_seed + instance,  # Different seed for each instance
                          maximisation=False,
                          verbose=True,
                          DoE_parameters={'criterion': "center", 'iterations': 1000})
    
    # Run the optimization
    optimizer(problem=problem)
    
    # Close the logger
    logger.close()
    
    # Extract convergence data
    result = extract_convergence_data(instance)
    if result:
        all_results.append(result)
    
    print(f"Optimization for instance {instance} completed.")
    print(f"Distance from optimum: {np.linalg.norm(problem.state.current_best.x - problem.optimum.x)}")
    print(f"Regret: {problem.state.current_best.y - problem.optimum.y}")

# Process results for plotting
if not all_results:
    print("No results to plot!")
    exit()

# Find the maximum number of evaluations across all runs
max_evals = max(result[0][-1] for result in all_results)

# Interpolate all results to have the same evaluation points
eval_points = np.linspace(1, max_evals, 100)
interpolated_values = []

for evals, best_so_far in all_results:
    # Interpolate to get values at common evaluation points
    interp_values = np.interp(eval_points, evals, best_so_far)
    interpolated_values.append(interp_values)

# Convert to numpy array for easier calculations
interpolated_values = np.array(interpolated_values)

# Calculate mean and standard deviation
mean_values = np.mean(interpolated_values, axis=0)
std_values = np.std(interpolated_values, axis=0)

# Create the convergence plot
plt.figure(figsize=(10, 6))
plt.plot(eval_points, mean_values, 'b-', label='Mean')
plt.fill_between(eval_points, 
                mean_values - std_values, 
                mean_values + std_values, 
                alpha=0.3, 
                color='b', 
                label='±1 std')

plt.xlabel('Function Evaluations')
plt.ylabel('Best Function Value')
plt.title(f'Convergence Plot for BBOB Function {problem_id} (Averaged over {len(all_results)} instances)')
plt.grid(True)
plt.legend()

# Save the plot
plot_path = os.path.join(results_dir, f"convergence_plot_function_{problem_id}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nConvergence plot saved to {plot_path}")

# Save the raw data
data_path = os.path.join(results_dir, f"convergence_data_function_{problem_id}.csv")
pd.DataFrame({
    'evaluations': eval_points,
    'mean': mean_values,
    'std': std_values
}).to_csv(data_path, index=False)

print(f"Raw data saved to {data_path}") 