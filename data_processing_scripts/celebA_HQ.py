import pandas as pd
import os
import PIL

mapping_path = '/home/rhu/big_downloads/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt'
attribute_path = '/home/rhu/big_downloads/list_attr_celeba.txt'

mappings  = pd.read_csv(mapping_path, sep="\s+")
attributes = pd.read_csv(attribute_path, sep="\s+")
joined_df = pd.merge(mappings,attributes, left_on='orig_file', right_on='name')
training_data = joined_df[['idx','Male']]
training_data['Male'] = training_data['Male'].apply(lambda x: 0 if x==-1 else 1)
training_data['idx'] = training_data['idx'].apply(lambda x: f'{str(x+1).zfill(5)}.jpg')
training_data = training_data.rename(columns={"idx": "file_name", "Male": "class"})
training_data.to_csv("../celebA_hq_gender.csv",index=0)
print(training_data['class'].sum())

