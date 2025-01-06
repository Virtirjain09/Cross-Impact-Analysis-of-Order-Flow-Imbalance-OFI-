import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def calculate_mlofi(group, levels=5):
    mlofi = pd.DataFrame(index=group.index)
    mlofi['symbol'] = group['symbol']
    for i in range(levels):
        bid_col = f'bid_sz_{i:02d}'
        ask_col = f'ask_sz_{i:02d}'
        mlofi[f'ofi_{i}'] = group[bid_col].diff().fillna(0) - group[ask_col].diff().fillna(0)
    return mlofi

def apply_pca(group):
    ofi_features = [f'ofi_{i}' for i in range(5)]
    pca = PCA(n_components=1)
    group_clean = group[ofi_features].dropna()
    if len(group_clean) > 0:
        group['ofi_pca'] = pd.Series(pca.fit_transform(group_clean).flatten(), index=group_clean.index)
    else:
        group['ofi_pca'] = np.nan
    return group

def calculate_price_changes(group):
    group['price_change'] = group['ofi_pca'].diff()
    group['future_price_change_1min'] = group['ofi_pca'].diff().shift(-60)  # Assuming 1-second intervals
    group['future_price_change_5min'] = group['ofi_pca'].diff().shift(-300)
    return group

def calculate_lagged_ofi(ofi_data):
    ofi_data['ofi_lagged'] = ofi_data.groupby('symbol')['ofi_pca'].shift(1)
    return ofi_data

