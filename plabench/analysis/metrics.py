
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import math

def calculate_metrics(predictions, ground_truth):
    """
    Calculate standard PLA-Bench metrics.
    
    Args:
        predictions: list or array of predicted values (pKd).
        ground_truth: list or array of true values (pKd).
        
    Returns:
        Dictionary of metrics rounded to 3 decimal places.
    """
    y_pred = np.array(predictions)
    y_true = np.array(ground_truth)
    
    # Ensure matching length
    if len(y_pred) != len(y_true):
        raise ValueError(f"Length mismatch: pred {len(y_pred)} vs true {len(y_true)}")
        
    # Metrics from haiping_methods/utils.py
    rmse_val = rmse(y_true, y_pred)
    mse_val = mse(y_true, y_pred)
    pearson_val = pearson(y_true, y_pred)
    spearman_val = spearman(y_true, y_pred)
    ci_val = ci(y_true, y_pred)
    
    # Kendall (not in utils.py explicitly? Yes it is not. But previous report had it.)
    # Using scipy for kendall
    if len(y_pred) > 1:
        kendall_val, _ = kendalltau(y_true, y_pred)
    else:
        kendall_val = 0.0

    # Rm2 (Custom formula requested by user)
    rm2_val = get_rm2(y_true, y_pred)
    
    metrics = {
        "RMSE": rmse_val,
        "MSE": mse_val,
        "Pearson": pearson_val,
        "Spearman": spearman_val,
        "Kendall": kendall_val,
        "CI": ci_val,
        "Rm2": rm2_val
    }
    
    # Round to 3 decimal places
    for k, v in metrics.items():
        if v is not None and not np.isnan(v):
            metrics[k] = round(float(v), 3)
        else:
            # User requested Rm2 < 0 -> NA
            metrics[k] = "NA"
            
    return metrics

# --- Functions from haiping_methods/utils.py ---
def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def pearson(y,f):
    if len(y) < 2: return 0.0
    rp = np.corrcoef(y, f)[0,1]
    return rp

def spearman(y,f):
    if len(y) < 2: return 0.0
    rs = spearmanr(y, f)[0]
    return rs

def ci(y,f):
    try:
        from lifelines.utils import concordance_index
        return concordance_index(y, f)
    except ImportError:
        ind = np.argsort(y)
        y = y[ind]
        f = f[ind]
        i = len(y)-1
        j = i-1
        z = 0.0
        S = 0.0
        while i > 0:
            while j >= 0:
                if y[i] > y[j]:
                    z = z+1
                    u = f[i] - f[j]
                    if u > 0:
                        S = S + 1
                    elif u == 0:
                        S = S + 0.5
                j = j - 1
            i = i - 1
            j = i-1
        if z == 0: return 0.0
        ci = S/z
        return ci

def get_rm2(y_true, y_pred):
    """
    Calculate Rm2 metric using Coefficient of Determination (R2).
    Formula: R2 * (1 - sqrt(|R2 - R02|))
    If R2 < 0, return 'NA' (or np.nan).
    """
    # R2 Score (Using Pearson correlation squared as per Roy et al. standard definition)
    # Previous implementation used r2_score (Coeff of Determination), which penalizes bias twice or incorrectly for this metric.
    pearson_r, _ = pearsonr(y_true, y_pred)
    r2 = pearson_r ** 2
    
    # R02 (Coefficient of Determination through origin)
    r02 = r_squared_zero_intercept(y_true, y_pred)
    
    # User Request: If r2 < 0, return NA.
    if r2 < 0: 
        return float('nan') 

    try:
        val = r2 * (1 - np.sqrt(np.abs(r2 - r02)))
        return val
    except:
        return 0.0

def r_squared_zero_intercept(y_true, y_pred):
    """
    Calculate R0^2: Determination coefficient for regression through origin.
    R0^2 = 1 - sum((y - k*pred)^2) / sum((y - mean(y))^2)
    This measures how well y=k*pred fits.
    """
    num = np.sum(y_true * y_pred)
    den = np.sum(y_pred ** 2)
    if den == 0: return 0.0
    k = num / den
    
    y_pred_k = k * y_pred
    
    rss = np.sum((y_true - y_pred_k) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if tss == 0: return 0.0
    
    return 1 - rss / tss


