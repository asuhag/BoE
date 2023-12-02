import pandas as pd

#data_pg1 = pd.read_excel('DataScientist_009749_Dataset.xlsx',sheet_name= 'Dataset 1 - General')
#data_pg2 = pd.read_excel('DataScientist_009749_Dataset.xlsx',sheet_name= 'Dataset 2 - Underwriting')
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
       'SCR coverage ratio 2016', 'SCR coverage ratio 2017', 'SCR coverage ratio 2018',
       'SCR coverage ratio 2019', 'SCR coverage ratio 2020', 'GWP 2016',
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
       'Net combined ratio.3', 'Net combined ratio.4']
    
    new_name_2 = ['Firm', 'GCI 2016',
       'GCI 2017', 'GCI 2018',
       'GCI 2019', 'GCI 2020',
     'NER 2016', 'NER 2017',
       'NER 2018', 'NER 2019', 'NER 2020',
       'NCR 2016', 'NCR 2017', 'NCR 2018',
       'NCR 2019', 'NCR 2020']

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