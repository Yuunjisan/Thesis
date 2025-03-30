__authors__ = ("Elena Raponi",
               "Ivan Olarte-Rodriguez")

r"""
This script runs Bayesian Optimization and creates convergence plots from the results.
"""

### -------------------------------------------------------------
### IMPORT LIBRARIES/REPOSITORIES
###---------------------------------------------------------------

# Algorithm import
from Algorithms import Vanilla_BO

# Standard libraries
import os
from pathlib import Path
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt

# IOH Experimenter libraries
try:
    from ioh import get_problem
    import ioh.iohcpp.logger as logger_lib
    from ioh.iohcpp.logger import Analyzer
    from ioh.iohcpp.logger.trigger import Each, ON_IMPROVEMENT, ALWAYS
except ModuleNotFoundError as e:
    print(e.args)
except Exception as e:
    print(e.args)

### ---------------------------------------------------------------
### LOGGER SETUP
### ---------------------------------------------------------------

# Define experiment name
experiment_folder = "bo-experiment"

# These are the triggers to set how to log data
triggers = [
    Each(1),  # Log after every evaluation (for detailed convergence curves)
    ON_IMPROVEMENT  # Log when there's an improvement
]

logger = Analyzer(
    triggers=triggers,
    root=os.getcwd(),
    folder_name=experiment_folder,
    algorithm_name="Vanilla BO",
    algorithm_info="Bo-Torch Implementation",
    additional_properties=[logger_lib.property.RAWYBEST],
    store_positions=True
)

# Modify the run_optimization function to better handle this issue

def run_optimization(problem_id=1, dimension=2, budget=None, n_runs=5):
    """
    Run Bayesian Optimization on the specified problem multiple times
    and return the data for convergence plots.
    """
    all_convergence_data = []
    
    # Create a separate logger for each run to avoid conflicts
    for run in range(1, n_runs+1):
        print(f"\nStarting run {run}/{n_runs} for problem {problem_id}, dimension {dimension}")
        
        try:
            # Set up problem instance with different instance ID for each run
            problem = get_problem(problem_id, instance=run, dimension=dimension)
            
            # Create a unique logger for this run to avoid potential conflicts
            run_logger = Analyzer(
                triggers=triggers,
                root=os.getcwd(),
                folder_name=f"{experiment_folder}/run_{run}",  # Separate subfolder
                algorithm_name=f"Vanilla BO Run {run}",
                algorithm_info="Bo-Torch Implementation",
                additional_properties=[logger_lib.property.RAWYBEST],
                store_positions=True
            )
            
            # Attach the run-specific logger
            problem.attach_logger(run_logger)
            
            # Set budget based on dimension if not provided
            if budget is None:
                actual_budget = min(200, 50*problem.meta_data.n_variables)
            else:
                actual_budget = budget
                
            # Set up the Vanilla BO with a different random seed for each run
            optimizer = Vanilla_BO(
                budget=actual_budget,
                n_DoE=3*problem.meta_data.n_variables,
                acquisition_function="expected_improvement",
                random_seed=42+run,  # Different seed for each run
                maximisation=False,
                verbose=True,
                DoE_parameters={'criterion': "center", 'iterations': 1000}
            )
            
            # Watch the optimizer with the run-specific logger
            run_logger.watch(optimizer, "acquistion_function_name")
            
            # Run optimization
            optimizer(problem=problem)
            
            # Get the raw data for this run
            evals = list(range(1, len(optimizer.f_evals) + 1))
            
            # Create list of best-so-far values
            best_so_far = []
            current_best = float('inf')
            for val in optimizer.f_evals:
                current_best = min(current_best, val)
                best_so_far.append(current_best)
                
            # Save run data
            run_data = {
                'run': run,
                'evals': evals,
                'best_so_far': best_so_far,
                'final_distance': norm(problem.state.current_best.x - problem.optimum.x),
                'final_regret': problem.state.current_best.y - problem.optimum.y
            }
            
            all_convergence_data.append(run_data)
            
            print(f"Run {run} completed successfully.")
            print(f"  - Final regret: {run_data['final_regret']:.6e}")
            print(f"  - Number of evaluations: {len(evals)}")
            
            # Close the run-specific logger
            run_logger.close()
            
        except Exception as e:
            print(f"ERROR: Run {run} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Verify we have all the expected runs
    if len(all_convergence_data) != n_runs:
        print(f"WARNING: Expected {n_runs} runs but only got {len(all_convergence_data)}")
    
    # Print a summary of collected runs
    print("\nRun summary:")
    for data in all_convergence_data:
        print(f"Run {data['run']}: {len(data['evals'])} evaluations, final regret: {data['final_regret']:.6e}")
    
    return all_convergence_data

def plot_convergence(convergence_data, problem_id, dimension, save_path=None):
    """
    Plot convergence curves from optimization runs with improved verification.
    """
    # First, let's print debug info to verify we have all runs
    print(f"\nDebug info - Number of runs: {len(convergence_data)}")
    
    # Create a mapping from run index to sequential index
    # This ensures we handle potential gaps in run numbers
    run_to_index = {}
    for i, run_data in enumerate(convergence_data):
        run_to_index[run_data['run']] = i
        print(f"Run {run_data['run']} (index {i}): {len(run_data['evals'])} evaluations, final value: {run_data['best_so_far'][-1]:.6e}")
    
    # Create the main convergence plot
    plt.figure(figsize=(12, 8))
    
    # Use a distinct color cycle with more colors
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(convergence_data))))
    if len(convergence_data) > 10:
        colors = plt.cm.jet(np.linspace(0, 1, len(convergence_data)))
    
    # Store handles for legend to ensure all runs are represented
    handles = []
    labels = []
    
    # Plot individual runs with distinct colors
    for i, run_data in enumerate(convergence_data):
        color_idx = i % len(colors)
        line, = plt.semilogy(
            run_data['evals'], 
            run_data['best_so_far'], 
            color=colors[color_idx],
            linewidth=1.5,
            alpha=0.7,
            label=f"Run {run_data['run']}"
        )
        handles.append(line)
        labels.append(f"Run {run_data['run']}")
    
    # Calculate and plot the mean across runs
    max_length = max(len(data['best_so_far']) for data in convergence_data)
    mean_values = []
    
    for i in range(max_length):
        values_at_i = [data['best_so_far'][i] if i < len(data['best_so_far']) else data['best_so_far'][-1] 
                      for data in convergence_data]
        mean_values.append(np.mean(values_at_i))
    
    # Plot mean with a very distinct style
    mean_line, = plt.semilogy(
        range(1, max_length + 1), 
        mean_values, 
        'k-', 
        linewidth=3,
        label='Mean'
    )
    handles.append(mean_line)
    labels.append('Mean')
    
    # Customize plot
    plt.title(f"Convergence Plot - Problem {problem_id}, Dimension {dimension} ({len(convergence_data)} runs)")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Function Value (log scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Use our explicit handles and labels to ensure all runs appear in the legend
    if len(convergence_data) > 10:
        # For many runs, select a subset for the legend
        selected_indices = [0, len(convergence_data)//2, len(convergence_data)-1, len(handles)-1]  # First, middle, last, and mean
        plt.legend([handles[i] for i in selected_indices], 
                  [labels[i] for i in selected_indices], 
                  loc='best')
    else:
        plt.legend(handles, labels, loc='best')
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    else:
        plt.show()
    
    # Create an improved verification plot that shows all runs
    plt.figure(figsize=(10, 5))
    plt.title("Verification of All Runs")
    
    # Use sequential indices for x-axis to ensure clear visualization
    indices = list(range(len(convergence_data)))
    final_values = [data['best_so_far'][-1] for data in convergence_data]
    run_numbers = [data['run'] for data in convergence_data]
    
    # Plot using sequential indices
    plt.semilogy(indices, final_values, 'bo', markersize=8)
    
    # Add run number labels
    for i, (run, value) in enumerate(zip(run_numbers, final_values)):
        plt.annotate(f"Run {run}", (i, value), textcoords="offset points", 
                   xytext=(0,10), ha='center')
    
    # Set x-axis ticks to show indices
    plt.xticks(indices, [f"{i}\n(Run {run})" for i, run in enumerate(run_numbers)])
    plt.xlabel("Run Index (Run Number)")
    plt.ylabel("Final Best Value (log scale)")
    plt.grid(True)
    
    # Save diagnostic plot
    if save_path:
        diag_path = str(save_path).replace('.png', '_verification.png')
        plt.savefig(diag_path, dpi=300, bbox_inches='tight')
        print(f"Verification plot saved to {diag_path}")
    
    # Summary statistics
    final_regrets = [data['final_regret'] for data in convergence_data]
    final_distances = [data['final_distance'] for data in convergence_data]
    
    print("\nSummary Statistics:")
    print(f"Mean final regret: {np.mean(final_regrets):.6e} ± {np.std(final_regrets):.6e}")
    print(f"Mean distance from optimum: {np.mean(final_distances):.6e} ± {np.std(final_distances):.6e}")
    
    # Extra verification: print all run numbers
    print("\nRun numbers in the convergence_data:")
    print(run_numbers)
    
    return plt

if __name__ == "__main__":
    # Parameters
    problem_id = 1  # Sphere function
    dimension = 2
    n_runs = 5
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("convergence_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Run optimization and get convergence data
    convergence_data = run_optimization(
        problem_id=problem_id,
        dimension=dimension,
        n_runs=n_runs
    )
    
    # Plot and save convergence curves
    plot_convergence(
        convergence_data,
        problem_id,
        dimension,
        save_path=plots_dir / f"problem_{problem_id}_dim_{dimension}.png"
    )
    
    # Close logger
    logger.close()
    
    print("\nOptimization and plotting complete!")