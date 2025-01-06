import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_cross_impact_heatmap(cross_impact, filename='cross_impact_heatmap.png'):
    plt.figure(figsize=(15, 10))
    mask = np.isnan(cross_impact)
    sns.heatmap(cross_impact,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                mask=mask,
                center=0,
                vmin=-1,
                vmax=1)
    plt.title('Cross-Impact Analysis Between Stocks')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_ofi_trends(ofi_data_cleaned, filename='ofi_trends.png'):
    plt.figure(figsize=(15, 10))
    for symbol in ofi_data_cleaned['symbol'].unique():
        symbol_data = ofi_data_cleaned[ofi_data_cleaned['symbol'] == symbol]
        plt.plot(range(len(symbol_data)), symbol_data['ofi_pca'], label=symbol, alpha=0.7)
    plt.title('OFI Trends Across Stocks')
    plt.xlabel('Time Index')
    plt.ylabel('OFI PCA')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_time_horizon_heatmaps(ofi_data_cleaned):
    time_horizons = {
        'Contemporaneous': 'price_change',
        '1-min Future': 'future_price_change_1min',
        '5-min Future': 'future_price_change_5min'
    }
    for horizon_name, column in time_horizons.items():
        correlations = pd.DataFrame(index=ofi_data_cleaned['symbol'].unique(),
                                    columns=ofi_data_cleaned['symbol'].unique(),
                                    dtype=float)
        for symbol1 in correlations.index:
            for symbol2 in correlations.columns:
                data1 = ofi_data_cleaned[ofi_data_cleaned['symbol'] == symbol1]['ofi_pca']
                data2 = ofi_data_cleaned[ofi_data_cleaned['symbol'] == symbol2][column]
                min_length = min(len(data1), len(data2))
                if min_length > 0:
                    correlation = data1[:min_length].corr(data2[:min_length])
                    correlations.loc[symbol1, symbol2] = correlation if np.isfinite(correlation) else np.nan
        plt.figure(figsize=(15, 10))
        mask = np.isnan(correlations)
        sns.heatmap(correlations,
                    annot=True,
                    cmap='coolwarm',
                    fmt='.2f',
                    mask=mask,
                    center=0,
                    vmin=-1,
                    vmax=1)
        plt.title(f'Cross-Impact Analysis for {horizon_name}')
        plt.tight_layout()
        plt.savefig(f'cross_impact_{horizon_name.lower()}.png')
        plt.close()

