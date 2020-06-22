from datasets.dataset import ImageDatasetFromFile,load_image,Image
from os.path import join
import torch
"""
Needed: pandas data and property file...
"""
class ImageDatasetFromFile_DIF(ImageDatasetFromFile):
    def __init__(self, property_indicator , image_list, root_path,
                input_height=128, input_width=None, output_height=128, output_width=None,
                crop_height=None, crop_width=None, is_random_crop=False, is_mirror=True, is_gray=False):
        super(ImageDatasetFromFile_DIF, self).__init__(image_list, root_path,
                input_height, input_width, output_height, output_width,
                crop_height, crop_width, is_random_crop, is_mirror, is_gray)
        self.property_indicator = property_indicator

    def __getitem__(self, index):  
        img = load_image(join(self.root_path, self.image_filenames[index]),
                                  self.input_height, self.input_width, self.output_height, self.output_width,
                                  self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray)
        img = self.input_transform(img)
        label = bool(self.property_indicator[index])

        return img,label

    def __len__(self):
        return len(self.image_filenames)

class only_Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        old_size = image.size[:2]
        ratio = float(self.output_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        im = image.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        img = Image.new("RGB", (self.output_size, self.output_size))
        img.paste(im, ((self.output_size - new_size[0]) // 2,
                          (self.output_size - new_size[1]) // 2))
        return img
