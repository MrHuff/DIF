from PIL import Image, ImageOps
import os





if __name__ == '__main__':
    path = '/homes/rhu/data/data256x256/'
    new_sizes = [32,64,128]
    files = os.listdir(path)
    for n in new_sizes:
        new_path = f'/homes/rhu/data/data{n}x{n}/'
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for f in files:
            img = Image.open(path+f).resize((n, n), Image.BICUBIC)
            img.save(new_path+f)

