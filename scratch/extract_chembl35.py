import pandas as pd

df = pd.read_csv('/bmlfast/Lyuwei/0.Projects/PLABench/results/weighted_summary_chembl35.csv')

# Format model names
display_names = {
    'flowr_root': 'FLOWR.ROOT',
    'haiping': 'Haiping',
    'bapred': 'BAPred',
    'flowdock': 'FlowDock',
    'boltz2': 'Boltz2',
    'mfe': 'MFE',
    'deepdta': 'DeepDTA',
    'llf': 'LLF',
    'mixingdta': 'MixingDTA'
}

latex = """
\\begin{table*}[t]
\\caption{Performance comparison of different models on the ChEMBL35 dataset. Bold and underlined values indicate the best performance.}
\\label{tab:chembl35_summary}
\\begin{center}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{lccccccc}
\\toprule
Model & MSE $\\downarrow$ & RMSE $\\downarrow$ & Pearson $\\uparrow$ & Spearman $\\uparrow$ & Kendall $\\uparrow$ & $R_m^2 \\uparrow$ & CI $\\uparrow$ \\\\
\\midrule
"""

def format_val(val, is_best):
    if pd.isna(val) or val == 'NA':
        return "NA"
    if is_best:
        return f"\\underline{{\\textbf{{{val:.3f}}}}}"
    else:
        return f"{val:.3f}"

metrics = ['MSE', 'RMSE', 'Pearson', 'Spearman', 'Kendall', 'Rm2', 'CI']
directions = ['min', 'min', 'max', 'max', 'max', 'max', 'max']

best_vals = {}
for m, d in zip(metrics, directions):
    if d == 'min':
        best_vals[m] = df[m].min()
    else:
        best_vals[m] = df[m].max()

for _, row in df.iterrows():
    model = row['Model']
    disp_name = display_names.get(model, model)
    
    formatted_vals = []
    for m in metrics:
        val = row[m]
        is_best = (val == best_vals[m])
        formatted_vals.append(format_val(val, is_best))
        
    latex += f"{disp_name} & {' & '.join(formatted_vals)} \\\\\n"

latex += """\\bottomrule
\\end{tabular}
}
\\end{center}
\\end{table*}
"""

print(latex)
