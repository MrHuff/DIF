import pandas as pd

csvs = [
    '/homes/rhu/data/celebA_hq_gender.csv',
    '/homes/rhu/data/fashion_price_class.csv',
    '/homes/rhu/data/mnist_3_8.csv',
    '/homes/rhu/data/covid_19_sick.csv'
]

for f in csvs:
    df = pd.read_csv(f)
    print('P: ',(df['class']==0).sum())
    print('Q: ',(df['class']==1).sum())