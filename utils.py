import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 

def convert_columns_to_numeric(df, exclude_col='Firm'):
    """
    Convert all columns in the DataFrame to numeric types, except for the specified exclude column.

    """
    for col in df.columns:
        if col != exclude_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def get_data(filepath,sheetname):
    """
    Read data from file and sheet. Rename columns for ease of query

    """

    current_name_1 = ['Unnamed: 0', 'NWP (£m) ', 'NWP (£m) .1', 'NWP (£m) .2', 'NWP (£m) .3',
       'NWP (£m) .4',
       'SCR coverage ratio', 'SCR coverage ratio.1', 'SCR coverage ratio.2',
       'SCR coverage ratio.3', 'SCR coverage ratio.4', 'GWP (£m)',
       'GWP (£m).1', 'GWP (£m).2', 'GWP (£m).3', 'GWP (£m).4', 
       'Excess of assets over liabilities (£m) [= equity]',
       'Excess of assets over liabilities (£m) [= equity].1',
       'Excess of assets over liabilities (£m) [= equity].2',
       'Excess of assets over liabilities (£m) [= equity].3',
       'Excess of assets over liabilities (£m) [= equity].4']
    
    new_name_1 = ['Firm', 'NWP 2016', 'NWP 2017', 'NWP 2018', 'NWP 2019',
       'NWP 2020',
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
    

def replace_outliers_with_mad(df, special_cols=None, threshold=3.5):
    """
    Replace outliers in all numeric columns of a DataFrame, with the median.
    For specified columns, replace outliers with the mean.
    Outliers are determined based on Median Absolute Deviation (MAD).

    special_cols (dict): A dictionary where keys are column names and values are 'mean' or 'median' for replacement.
    threshold (float): The threshold value for determining outliers based on the modified Z-score.
    """
    df_copy = df.copy()
    special_cols = special_cols or {}

    # Identifying numeric columns
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        # Calculate the median and the mean
        col_median = df_copy[col].median()
        col_mean = df_copy[col].mean()

        # Calculating the Median Absolute Deviation (MAD)
        mad = np.median(np.abs(df_copy[col] - col_median))

        # If MAD is zero (constant column), continue to the next column
        if mad == 0:
            continue

        # Calculating the modified Z-score
        modified_z_score = 0.6745 * (df_copy[col] - col_median) / mad

        # Identifying outliers
        outliers = np.abs(modified_z_score) > threshold

        # Replacing outliers with the median or mean based on special_cols
        if col in special_cols and special_cols[col] == 'mean':
            df_copy.loc[outliers, col] = col_mean
        else:
            df_copy.loc[outliers, col] = col_median

    return df_copy

def plot_top_firms_with_average(df, top_firms, ratio_column, title, x_label, y_label):
    """
    Function to plot top firms based on a specific ratio along with the overall average of that ratio.
    """
    
    # Calculate the overall average ratio
    overall_avg_ratio = df[ratio_column].mean()

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Firm', y=ratio_column, data=top_firms, palette='Set2')
    plt.axhline(overall_avg_ratio, color='red', linestyle='--', label=f'Overall Average Ratio = {overall_avg_ratio:.2f}')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
