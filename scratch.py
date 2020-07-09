# TEST COMMIT
import pandas as pd

df = pd.read_csv("fashion_price_class.csv")
df['file_name'] = df['file_name'].apply(lambda x: str(x)+'.jpg')
df.to_csv("fashion_price_class.csv",index=0)
