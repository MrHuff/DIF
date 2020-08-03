from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))
def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

if __name__ == '__main__':
    new_sizes = [128,256]
    path = f'/home/rhu/Documents/covid_dataset/'
    files = os.listdir(path)
    for n in new_sizes:
        new_path = f'/home/rhu/Documents/covid_dataset_{n}x{n}/'
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for f in files:
            img = Image.open(path+f)
            img = crop_max_square(img)
            img = img.resize((n, n), Image.BICUBIC)
            img.save(new_path+f)

