import os
import pandas as pd
from PIL import Image

class_A = [3,4]
class_B = [8,9]
raw_data_path_training = '/home/rhu/Downloads/mnist_png/training/'
raw_data_path_testing = '/home/rhu/Downloads/mnist_png/testing/'

def move_files(files,path,new_path,class_):
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for f in files:
        im = Image.open(path+f)
        im.save(new_path + str(class_)+'_'+f)

if __name__ == '__main__':
    for a,b in zip(class_A,class_B):
        data_set_path = f'/home/rhu/Downloads/mnist_{a}_{b}/'
        train_a = os.listdir(raw_data_path_training+str(a))
        test_a = os.listdir(raw_data_path_testing+str(a))
        train_b = os.listdir(raw_data_path_training+str(b))
        test_b = os.listdir(raw_data_path_testing+str(b))

        move_files(train_a,raw_data_path_training+str(a)+'/',data_set_path,a)
        move_files(test_a,raw_data_path_testing+str(a)+'/',data_set_path,a)
        move_files(train_b,raw_data_path_training+str(b)+'/',data_set_path,b)
        move_files(test_b,raw_data_path_testing+str(b)+'/',data_set_path,b)

        a_data = [str(a)+'_'+el for el in train_a + test_a]
        b_data = [str(b)+'_'+el for el in train_b + test_b]
        df = pd.DataFrame(a_data+b_data,columns=['file_name'])
        df['class'] = df['file_name'].apply(lambda x: 1 if x[0]==str(b) else 0)
        df = df.sample(frac=1)
        df.to_csv(f"../mnist_{a}_{b}.csv",index=0)