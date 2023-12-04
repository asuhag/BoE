import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 

def convert_columns_to_numeric(df, exclude_col='Firm'):
    """
    Convert all columns in the DataFrame to numeric types, except for the specified exclude column.

    :param df: Pandas DataFrame
    :param exclude_col: Column name to be excluded from conversion (default is 'Firm')
    :return: DataFrame with converted columns
    """
    for col in df.columns:
        if col != exclude_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def get_data(filepath,sheetname):

    current_name_1 = ['Unnamed: 0', 'NWP (£m) ', 'NWP (£m) .1', 'NWP (£m) .2', 'NWP (£m) .3',
       'NWP (£m) .4', 'SCR (£m)', 'SCR (£m).1', 'SCR (£m).2', 'SCR (£m).3',
       'SCR (£m).4', 'EoF for SCR (£m)', 'EoF for SCR (£m).1',
       'EoF for SCR (£m).2', 'EoF for SCR (£m).3', 'EoF for SCR (£m).4',
       'SCR coverage ratio', 'SCR coverage ratio.1', 'SCR coverage ratio.2',
       'SCR coverage ratio.3', 'SCR coverage ratio.4', 'GWP (£m)',
       'GWP (£m).1', 'GWP (£m).2', 'GWP (£m).3', 'GWP (£m).4', 
       'Excess of assets over liabilities (£m) [= equity]',
       'Excess of assets over liabilities (£m) [= equity].1',
       'Excess of assets over liabilities (£m) [= equity].2',
       'Excess of assets over liabilities (£m) [= equity].3',
       'Excess of assets over liabilities (£m) [= equity].4']
    
    new_name_1 = ['Firm', 'NWP 2016', 'NWP 2017', 'NWP 2018', 'NWP 2019',
       'NWP 2020', 'SCR 2016', 'SCR 2017', 'SCR 2018', 'SCR 2019',
       'SCR 2020', 'EoF for SCR 2016', 'EoF for SCR 2017',
       'EoF for SCR 2018', 'EoF for SCR 2019', 'EoF for SCR 2020',
       'SCR cr 2016', 'SCR cr 2017', 'SCR cr 2018',
       'SCR cr 2019', 'SCR cr 2020', 'GWP 2016',
       'GWP 2017', 'GWP 2018', 'GWP 2019', 'GWP 2020', 
       'EOAOL 2016',
       'EOAOL 2017',
       'EOAOL 2018',
       'EOAOL 2019',
       'EOAOL 2020']
    
    current_name_2 = ['Unnamed: 0', 'Gross claims incurred (£m)',
       'Gross claims incurred (£m).1', 'Gross claims incurred (£m).2',
       'Gross claims incurred (£m).3', 'Gross claims incurred (£m).4',
     'Net expense ratio', 'Net expense ratio.1',
       'Net expense ratio.2', 'Net expense ratio.3', 'Net expense ratio.4',
       'Net combined ratio', 'Net combined ratio.1', 'Net combined ratio.2',
       'Net combined ratio.3', 'Net combined ratio.4','Net BEL (inc. TPs as a whole, pre-TMTP) (£m)',
       'Net BEL (inc. TPs as a whole, pre-TMTP) (£m).1',
       'Net BEL (inc. TPs as a whole, pre-TMTP) (£m).2',
       'Net BEL (inc. TPs as a whole, pre-TMTP) (£m).3',
       'Net BEL (inc. TPs as a whole, pre-TMTP) (£m).4',]
    
    new_name_2 = ['Firm', 'GCI 2016',
       'GCI 2017', 'GCI 2018',
       'GCI 2019', 'GCI 2020',
     'NER 2016', 'NER 2017',
       'NER 2018', 'NER 2019', 'NER 2020',
       'NCR 2016', 'NCR 2017', 'NCR 2018',
       'NCR 2019', 'NCR 2020',
       'NBEL 2016', 'NBEL 2017', 'NBEL 2018', 'NBEL 2019', 'NBEL 2020']

    # mapper for first dataset - sheet 1 
    mapper_1 = dict(zip(current_name_1,new_name_1))

    # mapper for second dataset - sheet 2 
    mapper_2 = dict(zip(current_name_2, new_name_2))

    if sheetname == 'Dataset 1 - General':
        df = pd.read_excel(filepath, sheetname)
        df.rename(columns=mapper_1, errors="raise", inplace=True)
        return df[new_name_1]
    
    else:
        df = pd.read_excel(filepath, sheetname)
        df.rename(columns=mapper_2, errors="raise", inplace=True)
        return df[new_name_2]
    
def replace_outliers_with_median_except(df, exclude_col, threshold=3):
    """
    Replace outliers in all numeric columns of a DataFrame, except for a specified column, with the median of the respective column.

    :param df: Pandas DataFrame
    :param exclude_col: Column name to exclude from outlier processing
    :param threshold: Z-score value to determine outliers (default is 3)
    :return: inplace DataFrame with outliers replaced by median values
    """
    numeric_cols = [col for col in df.columns if col != exclude_col]

    for col in numeric_cols:
        col_median = df[col].median()
        col_z_score = np.abs((df[col] - df[col].mean()) / df[col].std())
        df.loc[col_z_score > threshold, col] = col_median

    return df

def replace_outliers_with_median(df, exclude_col = 'Firm', threshold=3):
    """
    Replace outliers in all numeric columns of a DataFrame, except for a specified column, with the median of the respective column.

    :param df: Pandas DataFrame
    :param exclude_col: Column name to exclude from outlier processing
    :param threshold: Z-score value to determine outliers (default is 3)
    :return: DataFrame with outliers replaced by median values
    """
    # Creating a copy of the DataFrame to avoid modifying the original one
    df_copy = df.copy()

    # Identifying numeric columns except the exclude_col
    numeric_cols = [col for col in df_copy.columns if df_copy[col].dtype.kind in 'biufc' and col != exclude_col]

    # Iterating through each numeric column
    for col in numeric_cols:
        # Calculating the median, mean, and standard deviation for the column
        col_median = df_copy[col].median()
        col_mean = df_copy[col].mean()
        col_std = df_copy[col].std()

        # Avoid division by zero in case of constant column
        if col_std == 0:
            continue

        # Calculating the Z-score for each value in the column
        col_z_score = np.abs((df_copy[col] - col_mean) / col_std)

        # Identifying outliers
        outliers = col_z_score > threshold

        # Replacing outliers with the median value
        df_copy.loc[outliers, col] = col_median

    return df_copy

def replace_outliers_with_mad(df, exclude_col = 'Firm', threshold=3.5):
    """
    Replace outliers in all numeric columns of a DataFrame, except for a specified column, with the median of the respective column.
    Outliers are determined based on Median Absolute Deviation (MAD).

    :param df: Pandas DataFrame
    :param exclude_col: Column name to exclude from outlier processing
    :param threshold: Threshold value to determine outliers based on MAD (default is 3.5)
    :return: DataFrame with outliers replaced by median values
    """
    # Creating a copy of the DataFrame to avoid modifying the original one
    df_copy = df.copy()

    # Identifying numeric columns except the exclude_col
    numeric_cols = [col for col in df_copy.columns if df_copy[col].dtype.kind in 'biufc' and col != exclude_col]

    # Iterating through each numeric column
    for col in numeric_cols:
        # Calculating the median for the column
        col_median = df_copy[col].median()

        # Calculating the Median Absolute Deviation (MAD)
        mad = np.median(np.abs(df_copy[col] - col_median))

        # If MAD is zero (constant column), continue to the next column
        if mad == 0:
            continue

        # Calculating the modified Z-score
        modified_z_score = 0.6745 * (df_copy[col] - col_median) / mad

        # Identifying outliers
        outliers = np.abs(modified_z_score) > threshold

        # Replacing outliers with the median value
        df_copy.loc[outliers, col] = col_median

    return df_copy

def visualize_outliers(df, cols):
    """
    Visualize outliers in specified columns of a DataFrame using boxplots.

    :param df: Pandas DataFrame
    :param cols: List of column names to visualize for outliers
    """
    num_plots = len(cols)
    plt.figure(figsize=(10, 4 * num_plots))

    for i, col in enumerate(cols, 1):
        plt.subplot(num_plots, 1, i)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()

def year_on_year_growth(df, metric_prefix):
    """
    Calculate year-on-year growth for a given metric and identify top 5 firms with the highest growth.
    Handles cases where the metric has zero entries and ensures numeric data types.

    :param df: Pandas DataFrame containing the data.
    :param metric_prefix: Prefix of the metric columns (e.g., 'SCR').
    :return: List of top 5 firms with the highest year-on-year growth.
    """
    # Filter columns related to the metric
    metric_cols = [col for col in df.columns if metric_prefix in col]
    metric_cols.sort()  # Ensure columns are in the correct year order

    # Calculate year-on-year growth
    epsilon = 1e-5  # Small value to avoid division by zero
    for i in range(len(metric_cols) - 1):
        # Avoid division by zero by adding epsilon to the denominator
        df[f'Growth {metric_cols[i+1]}'] = (df[metric_cols[i+1]] - df[metric_cols[i]]) / (df[metric_cols[i]] + epsilon)

    # Calculate average growth for each firm
    growth_cols = [col for col in df.columns if 'Growth' in col]
    df['Average Growth'] = df[growth_cols].mean(axis=1)

    # Ensure 'Average Growth' is treated as numeric
    df['Average Growth'] = pd.to_numeric(df['Average Growth'], errors='coerce')

    # Identify top 5 firms with the highest average growth
    top_firms = df.nlargest(5, 'Average Growth')['Firm']

    return top_firms.tolist()