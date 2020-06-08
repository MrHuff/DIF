from datasets.dataset import ImageDatasetFromFile,load_image
from os.path import join

"""
Needed: pandas data and property file...
"""
class ImageDatasetFromFile_DIF(ImageDatasetFromFile):
    def __init__(self, property_file , image_list, root_path,
                input_height=128, input_width=None, output_height=128, output_width=None,
                crop_height=None, crop_width=None, is_random_crop=False, is_mirror=True, is_gray=False):
        super(ImageDatasetFromFile_DIF, self).__init__(image_list, root_path,
                input_height, input_width, output_height, output_width,
                crop_height, crop_width, is_random_crop, is_mirror, is_gray)

    def __getitem__(self, index):  
        img = load_image(join(self.root_path, self.image_filenames[index]),
                                  self.input_height, self.input_width, self.output_height, self.output_width,
                                  self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray)
        img = self.input_transform(img)
        return img

    def __len__(self):
        return len(self.image_filenames)