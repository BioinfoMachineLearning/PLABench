import pandas as pd
import os

# Define the targets
models = ['deepdta', 'llf', 'mixingdta']
strategies = ['warm', 'cold_target', 'cold_drug']
datasets = ['davis', 'kiba']

# Map strategy and dataset to the corresponding string in the 'Dataset' column
dataset_map = {
    'warm': {'davis': 'davis_warm', 'kiba': 'kiba_warm'},
    'cold_target': {'davis': 'davis_cold_target', 'kiba': 'kiba_cold_target'},
    'cold_drug': {'davis': 'davis_cold_drug', 'kiba': 'kiba_cold_drug'}
}

# Read original CSV
df = pd.read_csv('/bmlfast/Lyuwei/0.Projects/PLABench/results/benchmark_summary.csv')

# Desired columns to extract and rename
# In CSV: MSE, RMSE, Pearson, Spearman, Rm2, CI
# We need it in order: MSE, RMSE, Pearson, Spearman, Rm2, CI
columns_to_extract = ['MSE', 'RMSE', 'Pearson', 'Spearman', 'Rm2', 'CI']

results = []

for strategy in strategies:
    for model in models:
        row = {'Strategy': 'Warm split' if strategy == 'warm' else ('Cold-target' if strategy == 'cold_target' else 'Cold-drug'), 'Model': model}
        
        # Davis
        davis_dataset = dataset_map[strategy]['davis']
        davis_row = df[(df['Model'] == model) & (df['Dataset'] == davis_dataset)]
        if not davis_row.empty:
            for col in columns_to_extract:
                val = davis_row[col].values[0]
                row[f'Davis_{col}'] = val
        else:
            for col in columns_to_extract:
                row[f'Davis_{col}'] = 'NA'
                
        # KIBA
        kiba_dataset = dataset_map[strategy]['kiba']
        kiba_row = df[(df['Model'] == model) & (df['Dataset'] == kiba_dataset)]
        if not kiba_row.empty:
            for col in columns_to_extract:
                val = kiba_row[col].values[0]
                row[f'KIBA_{col}'] = val
        else:
            for col in columns_to_extract:
                row[f'KIBA_{col}'] = 'NA'

        results.append(row)

res_df = pd.DataFrame(results)

# 1. Output the CSV
res_df.to_csv('/bmlfast/Lyuwei/0.Projects/PLABench/results/extracted_davis_kiba_metrics.csv', index=False)
print("CSV saved to results/extracted_davis_kiba_metrics.csv")

# 2. Output the LaTeX table
# Bold the best values according to their direction
# MSE (down), RMSE (down), Pearson (up), Spearman (up), Rm2 (up), CI (up)

latex = """
\\begin{table*}[t]
\\caption{Performance comparison of different models on Davis and KIBA datasets under various splitting strategies. Bold values indicate the best performance in each category.}
\\label{tab:davis_kiba_split}
\\begin{center}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{llcccccccccccc}
\\toprule
 &  & \\multicolumn{6}{c}{Davis} & \\multicolumn{6}{c}{KIBA} \\\\
\\cmidrule(lr){3-8} \\cmidrule(lr){9-14}
Strategy & Model & MSE $\\downarrow$ & RMSE $\\downarrow$ & Pearson $\\uparrow$ & Spearman $\\uparrow$ & $R_m^2 \\uparrow$ & CI $\\uparrow$ & MSE $\\downarrow$ & RMSE $\\downarrow$ & Pearson $\\uparrow$ & Spearman $\\uparrow$ & $R_m^2 \\uparrow$ & CI $\\uparrow$ \\\\
\\midrule
"""

def format_val(val, is_best):
    if is_best:
        return f"\\textbf{{{val:.3f}}}"
    else:
        return f"{val:.3f}"

for i, strategy in enumerate(strategies):
    # Find best values for this strategy
    strat_name = 'Warm split' if strategy == 'warm' else ('Cold-target' if strategy == 'cold_target' else 'Cold-drug')
    strat_rows = [r for r in results if r['Strategy'] == strat_name]
    
    best_davis = {'MSE': min([r['Davis_MSE'] for r in strat_rows]), 'RMSE': min([r['Davis_RMSE'] for r in strat_rows]),
                  'Pearson': max([r['Davis_Pearson'] for r in strat_rows]), 'Spearman': max([r['Davis_Spearman'] for r in strat_rows]),
                  'Rm2': max([r['Davis_Rm2'] for r in strat_rows]), 'CI': max([r['Davis_CI'] for r in strat_rows])}
                  
    best_kiba = {'MSE': min([r['KIBA_MSE'] for r in strat_rows]), 'RMSE': min([r['KIBA_RMSE'] for r in strat_rows]),
                  'Pearson': max([r['KIBA_Pearson'] for r in strat_rows]), 'Spearman': max([r['KIBA_Spearman'] for r in strat_rows]),
                  'Rm2': max([r['KIBA_Rm2'] for r in strat_rows]), 'CI': max([r['KIBA_CI'] for r in strat_rows])}

    for j, row in enumerate(strat_rows):
        strat_str = f"\\multirow{{3}}{{*}}{{{row['Strategy']}}}" if j == 0 else ""
        
        # Format Davis
        d_mse = format_val(row['Davis_MSE'], row['Davis_MSE'] == best_davis['MSE'])
        d_rmse = format_val(row['Davis_RMSE'], row['Davis_RMSE'] == best_davis['RMSE'])
        d_pearson = format_val(row['Davis_Pearson'], row['Davis_Pearson'] == best_davis['Pearson'])
        d_spearman = format_val(row['Davis_Spearman'], row['Davis_Spearman'] == best_davis['Spearman'])
        d_rm2 = format_val(row['Davis_Rm2'], row['Davis_Rm2'] == best_davis['Rm2'])
        d_ci = format_val(row['Davis_CI'], row['Davis_CI'] == best_davis['CI'])
        
        # Format KIBA
        k_mse = format_val(row['KIBA_MSE'], row['KIBA_MSE'] == best_kiba['MSE'])
        k_rmse = format_val(row['KIBA_RMSE'], row['KIBA_RMSE'] == best_kiba['RMSE'])
        k_pearson = format_val(row['KIBA_Pearson'], row['KIBA_Pearson'] == best_kiba['Pearson'])
        k_spearman = format_val(row['KIBA_Spearman'], row['KIBA_Spearman'] == best_kiba['Spearman'])
        k_rm2 = format_val(row['KIBA_Rm2'], row['KIBA_Rm2'] == best_kiba['Rm2'])
        k_ci = format_val(row['KIBA_CI'], row['KIBA_CI'] == best_kiba['CI'])
        
        model_name = "DeepDTA" if row['Model'] == 'deepdta' else ("LLF" if row['Model'] == 'llf' else "MixingDTA")

        latex += f"{strat_str} & {model_name} & {d_mse} & {d_rmse} & {d_pearson} & {d_spearman} & {d_rm2} & {d_ci} & {k_mse} & {k_rmse} & {k_pearson} & {k_spearman} & {k_rm2} & {k_ci} \\\\\n"
        
    latex += "\\midrule\n" if i < 2 else ""

latex += """\\bottomrule
\\end{tabular}
}
\\end{center}
\\end{table*}
"""

print("\nLaTeX Code:\n")
print(latex)
