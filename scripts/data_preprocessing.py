import pandas as pd
import numpy as np

def preprocess_data(data):
    data.dropna(inplace=True)
    data['timestamp'] = pd.to_datetime(data['ts_recv'])
    data.sort_values(by=['symbol', 'timestamp'], inplace=True)
    data.drop_duplicates(inplace=True)
    return data

def clean_ofi_data(ofi_data):
    important_columns = ['symbol', 'ofi_pca', 'price_change', 'future_price_change_1min', 'future_price_change_5min', 'ofi_lagged']
    numeric_columns = [col for col in important_columns if col != 'symbol']
    ofi_data_cleaned = ofi_data[important_columns].copy()
    ofi_data_cleaned[numeric_columns] = ofi_data_cleaned[numeric_columns].replace([np.inf, -np.inf], np.nan)
    ofi_data_cleaned = ofi_data_cleaned.dropna()
    return ofi_data_cleaned
