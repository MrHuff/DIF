import torch
from datasets.dataset_DIF import *
import pandas as pd
from main import load_model
from torchvision.utils import save_image

def dataloader_train_test(opt):
    data = pd.read_csv(opt.class_indicator_file)
    train_list = data['file_name'].values.tolist()[:opt.trainsize]
    train_property_indicator = data['class'].values.tolist()[:opt.trainsize]

    test_list = data['file_name'].values.tolist()[opt.trainsize:-1]
    test_property_indicator = data['class'].values.tolist()[opt.trainsize:-1]

    # swap out the train files

    assert len(train_list) > 0

    train_set = ImageDatasetFromFile_DIF(train_property_indicator, train_list, opt.dataroot, input_height=None,
                                         crop_height=None, output_height=opt.output_height, is_mirror=False)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True,
                                                    num_workers=1)

    test_set = ImageDatasetFromFile_DIF(test_property_indicator, test_list, opt.dataroot, input_height=None,
                                         crop_height=None, output_height=opt.output_height, is_mirror=False)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batchSize, shuffle=True,
                                                    num_workers=1)

    return train_data_loader,test_data_loader

def get_fake_images(model,device,n):
    with torch.no_grad():
        return model.sample_fake_eval(n,device)

def get_latents(model,real_images):
    with torch.no_grad():
        return model.get_latents(real_images)

def generate_image(model,z):
    with torch.no_grad():
        return model.decode(z)

def save_generated_images(image_tensor,path):
    save_image(image_tensor,path+'.jpg')

def generate_all_latents(dataloader,model):



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
