import pandas as pd
import os

n=64
data_path = f'/home/haaluser/phd_projects/data/fashion_{n}x{n}/'
file_names = os.listdir(data_path)
file_names = [el.split('.')[0] for el in file_names]
df = pd.read_csv('/home/haaluser/phd_projects/neuralnet-inference/NNINF/utils_1/picture_data_top_full_bigger.csv', sep=",")
df.drop_duplicates(subset="article_id",
                   keep=False, inplace=True)
df = df[df['article_id'].isin(file_names)]
df = df[['article_id','corporate_brand_price']]
median_price = df['corporate_brand_price'].median()
df['upper'] = df['corporate_brand_price'].apply(lambda x: 1 if x>=median_price else 0)
df = df[['article_id','upper']]
training_data = df.rename(columns={"article_id": "file_name", "upper": "class"})
training_data.to_csv("../fashion_price_class.csv",index=0)


