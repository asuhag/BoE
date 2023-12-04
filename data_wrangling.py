import pandas as pd
from explore_datafile import *

page_1 = get_data('DataScientist_009749_Dataset.xlsx', 'Dataset 1 - General')

print(page_1.head())
print(page_1.columns)

page_2 = get_data('DataScientist_009749_Dataset.xlsx', 'Dataset 2 - Underwriting')

print(page_2.head())
print(page_2.columns)

merged_df = pd.merge(left=page_1, right= page_2, left_on='Firm', right_on='Firm', how='left')
merged_df.drop(index = 0, inplace = True)
#merged_df.to_csv('data.csv', index = False)

convert_columns_to_numeric(merged_df)
new_df = replace_outliers_with_mad(merged_df)

# Top 5 companies with high liquidity risk 
new_df['Avg SCR cr'] = new_df.filter(regex='SCR cr').mean(axis = 1)
top_5_lowest_scr = new_df.sort_values(by='Avg SCR cr').head(5)[['Firm','Avg SCR cr']]
plot_top_firms_with_average(new_df, top_5_lowest_scr, 'Avg SCR cr', 'Liquidity risk - Top 5 Firms with Lowest Average SCR coverage ratio', 
                            'Firm', 'Average SCR coverage ratio') 

# Top 5 companies with high strategic risk
new_df['Avg NBEL'] = new_df.filter(regex='NBEL').mean(axis = 1)
top_5_highest_nbel = new_df.sort_values(by='Avg NBEL', ascending=False).head(5)[['Firm','Avg NBEL']]

plot_top_firms_with_average(new_df, top_5_highest_nbel, 'Avg NBEL', 'Strategic risk - Top 5 Firms with Highest Average NBEL', 
                            'Firm', 'Average NBEL Ratio') 

# Top 5 companies with high operation risk 
new_df['Avg NCR'] = new_df.filter(regex='NCR').mean(axis=1)
top_5_highest_ncr = new_df.sort_values(by='Avg NCR', ascending=False).head(5)[['Firm', 'Avg NCR']]
plot_top_firms_with_average(new_df, top_5_highest_ncr, 'Avg NCR', ' Operation risk - Top 5 Firms with Highest Average NCR', 
                            'Firm', 'Average NCR Ratio') 

# Top 5 companies with high strategic risk because of reinsurance dependence 
years = ['2016', '2017', '2018', '2019', '2020']

# Adding a small value to NWP to avoid division by zero
small_value = 1e-5
for year in years:
    gwp_col = f'GWP {year}'
    nwp_col = f'NWP {year}'
    ratio_col = f'GWP/NWP Ratio {year}'

    # Adding small value to NWP and ensuring NWP is not extremely small
    new_df[nwp_col] = new_df[nwp_col].apply(lambda x: x + small_value if x < small_value else x)

    # Calculating the ratio
    new_df[ratio_col] = new_df[gwp_col] / new_df[nwp_col]

# Replace 'inf' values with NaN
new_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Calculating the average GWP/NWP ratio across years
new_df['Avg GWP/NWP'] = new_df.filter(regex='GWP/NWP Ratio').mean(axis=1)

# Finding the top 5 firms with the highest average GWP/NWP ratio
top_5_highest_reinsurance = new_df.sort_values(by='Avg GWP/NWP', ascending=False).head(5)[['Firm', 'Avg GWP/NWP']]

print("Top 5 Firms with Highest Average GWP/NWP Ratio:")
print(top_5_highest_reinsurance)
plot_top_firms_with_average(new_df, top_5_highest_reinsurance, 'Avg GWP/NWP', 'Strategic risk - Top 5 Firms with Highest Dependence on Reinsurance', 
                            'Firm', 'Average GWP/NWP Ratio') 
