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

