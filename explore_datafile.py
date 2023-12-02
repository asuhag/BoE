import pandas as pd

data_pg1 = pd.read_excel('DataScientist_009749_Dataset.xlsx',sheet_name= 'Dataset 1 - General')
data_pg2 = pd.read_excel('DataScientist_009749_Dataset.xlsx',sheet_name= 'Dataset 2 - Underwriting')

print(data_pg1.head())
print(data_pg2.head())