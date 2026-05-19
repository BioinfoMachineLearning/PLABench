import pandas as pd
import os

df = pd.read_csv('/bmlfast/Lyuwei/0.Projects/PLABench/results/benchmark_summary.csv')

models = ['flowr_root', 'haiping', 'bapred', 'flowdock', 'boltz2', 'mfe', 'deepdta', 'llf', 'mixingdta']

metrics = ['MSE', 'RMSE', 'Pearson', 'Spearman', 'Rm2', 'CI']
target_types = ['l1000', 'l3000']

def find_best_dataset(model, target_type, df):
    sub_df = df[(df['Model'] == model) & (df['Dataset'].str.contains(f'casp16_{target_type}'))]
    sub_df = sub_df[~sub_df['Dataset'].str.contains('stage2')]
    sub_df = sub_df[~sub_df['Dataset'].str.contains('v[1-6]', regex=True)]
    if sub_df.empty: return None
    datasets = sub_df['Dataset'].tolist()
    af3_datasets = [d for d in datasets if 'af3' in d]
    if af3_datasets: return af3_datasets[0]
    stage1_datasets = [d for d in datasets if 'stage1' in d]
    if stage1_datasets: return stage1_datasets[0]
    datasets.sort(key=len)
    return datasets[0]

results = []

for model in models:
    # Use exact raw name
    row = {'Model': model}
    for t_idx, target_type in enumerate(target_types):
        prefix = 'L1000_' if target_type == 'l1000' else 'L3000_'
        dataset_name = find_best_dataset(model, target_type, df)
        if dataset_name:
            match_row = df[(df['Model'] == model) & (df['Dataset'] == dataset_name)]
            for m in metrics:
                val = match_row[m].values[0]
                row[prefix + m] = val
        else:
            for m in metrics:
                row[prefix + m] = 'NA'
    results.append(row)

res_df = pd.DataFrame(results)
res_df.to_csv('/bmlfast/Lyuwei/0.Projects/PLABench/results/casp16_stage1_full_metrics.csv', index=False)

latex = """
\\begin{table*}[t]
\\caption{Performance comparison of different models on CASP16 datasets (L1000 and L3000) for Stage 1. Models with multiple input conformations use AF3 inputs. Bold and underlined values indicate the best performance.}
\\label{tab:casp16_stage1}
\\begin{center}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{lcccccccccccc}
\\toprule
 & \\multicolumn{6}{c}{L1000} & \\multicolumn{6}{c}{L3000} \\\\
\\cmidrule(lr){2-7} \\cmidrule(lr){8-13}
Model & MSE $\\downarrow$ & RMSE $\\downarrow$ & Pearson $\\uparrow$ & Spearman $\\uparrow$ & $R_m^2 \\uparrow$ & CI $\\uparrow$ & MSE $\\downarrow$ & RMSE $\\downarrow$ & Pearson $\\uparrow$ & Spearman $\\uparrow$ & $R_m^2 \\uparrow$ & CI $\\uparrow$ \\\\
\\midrule
"""

def format_val(val, is_best):
    if pd.isna(val) or val == 'NA':
        return "NA"
    if is_best:
        return f"\\underline{{\\textbf{{{val:.3f}}}}}"
    else:
        return f"{val:.3f}"

best_vals = {}
for prefix in ['L1000_', 'L3000_']:
    mse_vals = [r[prefix+'MSE'] for r in results if r[prefix+'MSE'] != 'NA']
    rmse_vals = [r[prefix+'RMSE'] for r in results if r[prefix+'RMSE'] != 'NA']
    p_vals = [r[prefix+'Pearson'] for r in results if r[prefix+'Pearson'] != 'NA']
    s_vals = [r[prefix+'Spearman'] for r in results if r[prefix+'Spearman'] != 'NA']
    r_vals = [r[prefix+'Rm2'] for r in results if r[prefix+'Rm2'] != 'NA']
    c_vals = [r[prefix+'CI'] for r in results if r[prefix+'CI'] != 'NA']
    
    best_vals[prefix] = {
        'MSE': min(mse_vals) if mse_vals else None,
        'RMSE': min(rmse_vals) if rmse_vals else None,
        'Pearson': max(p_vals) if p_vals else None,
        'Spearman': max(s_vals) if s_vals else None,
        'Rm2': max(r_vals) if r_vals else None,
        'CI': max(c_vals) if c_vals else None
    }

for j, row in enumerate(results):
    l1_mse = format_val(row['L1000_MSE'], row['L1000_MSE'] == best_vals['L1000_']['MSE'])
    l1_rmse = format_val(row['L1000_RMSE'], row['L1000_RMSE'] == best_vals['L1000_']['RMSE'])
    l1_p = format_val(row['L1000_Pearson'], row['L1000_Pearson'] == best_vals['L1000_']['Pearson'])
    l1_s = format_val(row['L1000_Spearman'], row['L1000_Spearman'] == best_vals['L1000_']['Spearman'])
    l1_r = format_val(row['L1000_Rm2'], row['L1000_Rm2'] == best_vals['L1000_']['Rm2'])
    l1_c = format_val(row['L1000_CI'], row['L1000_CI'] == best_vals['L1000_']['CI'])
    
    l3_mse = format_val(row['L3000_MSE'], row['L3000_MSE'] == best_vals['L3000_']['MSE'])
    l3_rmse = format_val(row['L3000_RMSE'], row['L3000_RMSE'] == best_vals['L3000_']['RMSE'])
    l3_p = format_val(row['L3000_Pearson'], row['L3000_Pearson'] == best_vals['L3000_']['Pearson'])
    l3_s = format_val(row['L3000_Spearman'], row['L3000_Spearman'] == best_vals['L3000_']['Spearman'])
    l3_r = format_val(row['L3000_Rm2'], row['L3000_Rm2'] == best_vals['L3000_']['Rm2'])
    l3_c = format_val(row['L3000_CI'], row['L3000_CI'] == best_vals['L3000_']['CI'])
    
    # Escape underscores for LaTeX
    model_name = row['Model'].replace('_', '\\_')

    latex += f"{model_name} & {l1_mse} & {l1_rmse} & {l1_p} & {l1_s} & {l1_r} & {l1_c} & {l3_mse} & {l3_rmse} & {l3_p} & {l3_s} & {l3_r} & {l3_c} \\\\\n"

latex += """\\bottomrule
\\end{tabular}
}
\\end{center}
\\end{table*}
"""

print(latex)
