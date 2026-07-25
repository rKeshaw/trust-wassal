import pandas as pd

import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trust.utils.paths import RESULTS_DIR

# File paths for each budget
budgets = [20, 40, 60, 80]
_pneumo_dir = str(RESULTS_DIR / "inpaper" / "pneumoniamnist" / "classimb" / "rounds10")
filepaths = [
    _pneumo_dir + "/badge/{budget}/exp2/pneumoniamnist_2_badge_budget:{budget}_rounds:10_runs_exp2.csv",
    #_pneumo_dir + "/random/{budget}/exp2/pneumoniamnist_2_random_budget:{budget}_rounds:10_runs_exp2.csv",
    _pneumo_dir + "/coreset/{budget}/exp2/pneumoniamnist_2_coreset_budget:{budget}_rounds:10_runs_exp2.csv",
    _pneumo_dir + "/us/{budget}/exp2/pneumoniamnist_2_us_budget:{budget}_rounds:10_runs_exp2.csv",
    _pneumo_dir + "/WASSAL/{budget}/exp2/pneumoniamnist_2_WASSAL_WITHSOFT_budget:{budget}_rounds:10_runs_exp2.csv"
]

# Read data without headers and manually assign column names
column_names = ["Class0", "Class1", "Avg"]

# Create a 2x2 grid of subplots for ALcurve
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Plot for each budget in ALcurve
for i, budget in enumerate(budgets):
    # Generate file paths for the current budget
    current_filepaths = [filepath.format(budget=budget) for filepath in filepaths]
    
    # Read data for the current budget
    data = [pd.read_csv(filepath, header=None, names=column_names) for filepath in current_filepaths]
    
    # Extract the average data for each approach
    avg_data = [df['Avg'].values for df in data]
    
    # Plot the data for each approach in the corresponding subplot
    ax = axs[i // 2, i % 2]
    for avg in avg_data:
        ax.plot(avg, marker='o', linestyle='--')
    
    # Set plot labels and legend for each subplot
    ax.set_title(f'Accuracy for Budget {budget}')
    ax.set_xlabel('AL Rounds')
    ax.set_ylabel('Value')
    ax.legend(['Badge', 'Random', 'Coreset', 'Uncertainty Sampling', 'WASSAL Accuracy'])
    ax.grid(True)

# Adjust the layout and save the figure
plt.tight_layout()
plt.savefig('ALcurve.png')

# Create a new figure ALcurve1
fig1, axs1 = plt.subplots(2,2, figsize=(10, 6))

# Plot for each strategy in ALcurve1
for i, filepath in enumerate(filepaths):
    # Read data for each budget
    data = [pd.read_csv(filepath.format(budget=budget), header=None, names=column_names) for budget in budgets]
    
    # Extract the average data for each budget
    avg_data = [df['Avg'].values for df in data]
    
    # Plot the data for each budget in the corresponding subplot
    ax1 = axs1[i // 2, i % 2]
    for avg in avg_data:
        ax1.plot(avg, marker='o', linestyle='--')
    
    # Set plot labels and legend for each subplot
    # Determine the Strategy name from the filepath
    strategy_name = filepath.split('/')[-4]
    
    # Set the title for each subplot
    ax1.set_title(f'{strategy_name}')
    ax1.set_xlabel(f'Figure {i+1}')
    ax1.set_ylabel('Accuaracy')
    ax1.legend([f'Budget {budget}' for budget in budgets])
    ax1.grid(True)

# Adjust the layout and save the figure
plt.tight_layout()
plt.savefig('ALcurve1.png')


