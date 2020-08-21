import pandas as pd

csvs = [
    '/homes/anonymous/data/celebA_hq_gender.csv',
    '/homes/anonymous/data/fashion_price_class.csv',
    '/homes/anonymous/data/mnist_3_8.csv',
    '/homes/anonymous/data/covid_19_sick.csv'
]

for f in csvs:
    df = pd.read_csv(f)
    print('P: ',(df['class']==0).sum())
    print('Q: ',(df['class']==1).sum())