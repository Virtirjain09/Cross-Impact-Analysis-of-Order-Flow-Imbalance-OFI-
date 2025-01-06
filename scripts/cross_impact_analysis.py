import numpy as np
import pandas as pd
import statsmodels.api as sm

def run_regression(data, y_col, x_cols):
    df = data[[y_col] + x_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        print(f"No valid data for regression with y={y_col} and x={x_cols}")
        return None
    X = sm.add_constant(df[x_cols])
    y = df[y_col]
    model = sm.OLS(y, X).fit()
    return model

def cross_impact_analysis(data):
    symbols = data['symbol'].unique()
    impact_matrix = pd.DataFrame(index=symbols, columns=symbols, dtype=float)
    for symbol1 in symbols:
        for symbol2 in symbols:
            try:
                data1 = data[data['symbol'] == symbol1]['ofi_pca'].values
                data2 = data[data['symbol'] == symbol2]['price_change'].values
                min_length = min(len(data1), len(data2))
                if min_length > 0:
                    correlation = np.corrcoef(data1[:min_length], data2[:min_length])[0,1]
                    impact_matrix.loc[symbol1, symbol2] = correlation if np.isfinite(correlation) else np.nan
                else:
                    impact_matrix.loc[symbol1, symbol2] = np.nan
            except Exception as e:
                impact_matrix.loc[symbol1, symbol2] = np.nan
    return impact_matrix

