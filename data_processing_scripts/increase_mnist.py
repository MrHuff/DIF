from PIL import Image, ImageOps
import os


if __name__ == '__main__':
    new_sizes = [64]
    a=4
    b=9
    path = f'/home/rhu/Downloads/mnist_{a}_{b}/'
    files = os.listdir(path)
    for n in new_sizes:
        new_path = f'/home/rhu/Downloads/mnist_{a}_{b}_{n}x{n}/'
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for f in files:
            img = Image.open(path+f).resize((n, n), Image.BICUBIC)
            img.save(new_path+f)

