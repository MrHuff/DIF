from PIL import Image, ImageOps
import os
from datasets.dataset_DIF import only_Rescale
from tqdm import tqdm
from joblib import Parallel, delayed
def process(f,path):
    try:
        img = Image.open(path + f)
        img = t(img)
        save = f.split('.')[0] + '.jpg'
        img.save(new_path + save)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    path = '/home/haaluser/phd_projects/neuralnet-inference/NNINF/utils_1/raw_pictures_top_bigger/'
    new_sizes = [256,128,64]
    files = os.listdir(path)
    for n in new_sizes:
        t = only_Rescale(n)
        new_path = f'/home/haaluser/phd_projects/data/fashion_{n}x{n}/'
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        results = Parallel(n_jobs=6)(delayed(process)(f,path) for f in tqdm(files))
        # for f in files:
        #     process(f,path)

