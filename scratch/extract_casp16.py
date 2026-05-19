import pandas as pd

df = pd.read_csv('/bmlfast/Lyuwei/0.Projects/PLABench/results/benchmark_summary.csv')

# Filter for CASP16 datasets. Exclude stage2, finetune, v1, v4, v5, v6 to keep only the pure stage1 models.
casp16_rows = df[
    df['Dataset'].str.contains('casp16', na=False) & 
    ~df['Dataset'].str.contains('stage2', na=False) &
    ~df['Dataset'].str.contains('v[1-6]', regex=True) &
    ~df['Model'].str.contains('stage2')
]

methods = {}

for _, row in casp16_rows.iterrows():
    model = row['Model']
    dataset = row['Dataset']
    kendall = row['Kendall']
    n_targets = row['N']
    
    # parse l1000 or l3000
    target_type = None
    if 'l1000' in dataset:
        target_type = 'L1000'
    elif 'l3000' in dataset:
        target_type = 'L3000'
        
    if not target_type:
        continue
        
    # generate a clean method name
    # e.g. "flowdock (boltz2_stage1)" -> "flowdock (boltz2)"
    base_name = dataset.replace('_casp16_l1000', '').replace('_casp16_l3000', '').replace('_stage1', '').replace('haiping_', '').replace('flowr_root_', '')
    if base_name == '':
        method_name = model
    else:
        method_name = f"{model} ({base_name})"
        
    if method_name not in methods:
        methods[method_name] = {'L1000': None, 'L1000_N': 0, 'L3000': None, 'L3000_N': 0}
        
    methods[method_name][target_type] = kendall
    methods[method_name][f'{target_type}_N'] = n_targets

data = []
for m, vals in methods.items():
    l1000_k = vals['L1000']
    l3000_k = vals['L3000']
    
    # Calculate weighted
    weighted_k = None
    if pd.notna(l1000_k) and pd.notna(l3000_k):
        n1 = vals['L1000_N']
        n3 = vals['L3000_N']
        if n1 + n3 > 0:
            weighted_k = (l1000_k * n1 + l3000_k * n3) / (n1 + n3)
            
    data.append({
        'Method': m,
        'L1000': l1000_k,
        'L3000': l3000_k,
        'Weighted': weighted_k
    })

res_df = pd.DataFrame(data)
res_df = res_df.dropna(subset=['Weighted'])
res_df.to_csv('/bmlfast/Lyuwei/0.Projects/PLABench/results/casp16_stage1_kendall.csv', index=False)
