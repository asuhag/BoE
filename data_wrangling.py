import pandas as pd
from explore_datafile import get_data

page_1 = get_data('DataScientist_009749_Dataset.xlsx', 'Dataset 1 - General')

print(page_1.head())
print(page_1.columns)

page_2 = get_data('DataScientist_009749_Dataset.xlsx', 'Dataset 2 - Underwriting')

print(page_2.head())
print(page_2.columns)
