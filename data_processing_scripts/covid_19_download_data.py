import pandas as pd
import os
import json
import urllib.request
import numpy as np
import multiprocessing

def func(chunk):
    for i, row in chunk.iterrows():
        if not os.path.isfile(save_path+row['file_name']):
            urllib.request.urlretrieve(row['url'], save_path+row['file_name'])


if __name__ == '__main__':


    path  = '../../covid-19-xray-dataset/annotations/covid-only/'
    covid_json_files = os.listdir(path)
    covid_files = []

    for f in covid_json_files:
        read_json = path+f
        try:
            with open(read_json, 'r') as j:
                contents = json.loads(j.read())
                is_CT=False
                is_lateral = False
                for d in contents['annotations']:
                    if 'CT' in d['name']:
                        is_CT=True
                    if 'Lateral' in d['name']:
                        is_lateral = True
                if (not is_CT) and (not is_lateral):
                    covid_files.append([contents['image']['filename'],contents['image']['url'],1])
        except Exception as e:
            print(e)

    all_path  = '../../covid-19-xray-dataset/annotations/all-images/'
    all_json_files = os.listdir(all_path)
    print(len(all_json_files))
    for f in all_json_files:
        read_json = all_path+f
        try:
            with open(read_json, 'r') as j:
                contents = json.loads(j.read())
                no_covid = False
                is_ct = False
                is_lateral = False

                for d in contents['annotations']:
                    if 'No Pneumonia (healthy)' in d['name']:
                        no_covid  = True
                    if 'CT' in d['name']:
                        is_CT = False
                    if 'Lateral' in d['name']:
                        is_lateral = True

                if (not is_lateral) and no_covid and (not is_CT):
                    covid_files.append([contents['image']['filename'], contents['image']['url'], 0])
        except Exception as e:
            print(e)
    manual_remove = ['00006651.jpeg','00006672.jpg','00006679.jpg','00006680.jpg'] #Remove unlabeled CT-scans

    df = pd.DataFrame(covid_files,columns=['file_name','url','class'])
    df[~df['file_name'].isin(manual_remove)][['file_name','class']].to_csv('covid_19_sick.csv',index=0)
    save_path = '../../../covid_dataset/'


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_cores = 4# leave one free to not freeze machine
    num_partitions = num_cores  # number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    pool.map(func, df_split)
    pool.close()
    pool.join()
