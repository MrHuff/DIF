import os
import pandas as pd
from PIL import Image

class_A = [0,1,2,3,4,5,6,7,8,9]
raw_data_path_training = '/home/rhu/Downloads/mnist_png/training/'
raw_data_path_testing = '/home/rhu/Downloads/mnist_png/testing/'

def move_files(files,path,new_path,class_):
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for f in files:
        im = Image.open(path+f)
        im.save(new_path + str(class_)+'_'+f)

if __name__ == '__main__':
    data_set_path = f'/home/rhu/Downloads/mnist_full/'
    for digit_class in class_A:
        train_a = os.listdir(raw_data_path_training+str(digit_class))
        test_a = os.listdir(raw_data_path_testing+str(digit_class))
        move_files(train_a,raw_data_path_training+str(digit_class)+'/',data_set_path,digit_class)
        move_files(test_a,raw_data_path_testing+str(digit_class)+'/',data_set_path,digit_class)
    files = os.listdir(data_set_path)
    df = pd.DataFrame(files,columns=['file_name'])
    for digit_class in class_A:
        df[f'bool_is_{digit_class}'] = df['file_name'].apply(lambda x: 1 if x[0]==str(digit_class) else 0)
    df = df.sample(frac=1)
    df.to_csv(f"../mnist_full.csv",index=False)