import torch
from datasets.dataset_DIF import *
import pandas as pd
from main import load_model
from torchvision.utils import save_image
import tqdm
from torch.cuda.amp import autocast,GradScaler
import os
from torchvision.utils import make_grid

def dataloader_train_test(opt):
    data = pd.read_csv(opt.class_indicator_file)
    train_list = data['file_name'].values.tolist()[:opt.trainsize]
    train_property_indicator = data['class'].values.tolist()[:opt.trainsize]

    test_list = data['file_name'].values.tolist()[opt.trainsize:-1]
    test_property_indicator = data['class'].values.tolist()[opt.trainsize:-1]

    # swap out the train files

    assert len(train_list) > 0

    train_set = ImageDatasetFromFile_DIF(train_property_indicator, train_list, opt.dataroot, input_height=None,
                                         crop_height=None, output_height=opt.output_height, is_mirror=False,is_gray=opt.cdim!=3)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True,
                                                    num_workers=opt.workers)

    test_set = ImageDatasetFromFile_DIF(test_property_indicator, test_list, opt.dataroot, input_height=None,
                                         crop_height=None, output_height=opt.output_height, is_mirror=False,is_gray=opt.cdim!=3)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batchSize, shuffle=True,
                                                    num_workers=opt.workers)

    return train_data_loader,test_data_loader

def get_fake_images(model,n):
    with torch.no_grad():
        return model.sample_fake_eval(n)

def save_images_individually(images, dir, folder, file_name):
    if not os.path.exists(dir+folder):
        os.makedirs(dir+folder)
    n = images.shape[0]
    for i in range(n):
        save_image(images[i, :, :, :], dir + folder + f'/{file_name}_{i}.jpg')

def save_images_group(images, dir, folder, file_name):
    if not os.path.exists(dir+folder):
        os.makedirs(dir+folder)
    save_image(images, dir + folder + f'/{file_name}.jpg')

def get_latents(model,real_images):
    with torch.no_grad():
        return model.get_latent(real_images)

def generate_image(model,z):
    with torch.no_grad():
        return model.decode(z)

def generate_all_latents(dataloader,model):
    _latents = []
    _class = []
    for iteration, (batch, c) in enumerate(tqdm.tqdm(dataloader)):
        with autocast():
            z = get_latents(model,batch.cuda())
        _latents.append(z.float())
        _class.append(c)
    _latents = torch.cat(_latents,dim=0)
    _class = torch.cat(_class,dim=0)
    return _latents,_class

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def feature_traverse(latent,id,C):
    trav = [C/10.*i-C for i in range(21)]
    l = torch.zeros_like(latent)
    l[id]=1
    container = []
    for z in trav:
        container.append(z*l+latent)
    return torch.stack(container,dim=0)


def feature_isolation(C, z_test, c_test, lasso_model, model,folder_path):
    x_test = z_test[~c_test, :]
    y_test = z_test[c_test, :]
    imgs_x = x_test[0:5, :]
    imgs_y = y_test[0:5, :]
    w = torch.abs(lasso_model.linear.weight)
    _,idx = torch.topk(w,5)
    list_id_x = []
    list_id_y = []
    for i in range(5):
        id = idx.squeeze()[i]
        list_x = []
        list_y = []
        for j in range(5):
            list_x.append(feature_traverse(imgs_x[j,:],id,C))
            list_y.append(feature_traverse(imgs_y[j,:],id,C))
        list_id_x.append(list_x)
        list_id_y.append(list_y)
    for i in range(5):
        for j in range(5):
            a = list_id_x[i][j]
            b = list_id_y[i][j]
            imgs_a = generate_image(model,a)
            save_images_group(imgs_a,folder_path,'feature_isolate_A',f'isolate_feature_{i}_pic_{j}')
            imgs_b = generate_image(model,b)
            save_images_group(imgs_b,folder_path,'feature_isolate_B',f'isolate_feature_{i}_pic_{j}')
            j+=1

def traverse(z_test, c_test, model,folder_path):
    x_test = z_test[~c_test, :]
    y_test = z_test[c_test, :]
    imgs_x = x_test[10:20, :]
    imgs_y = y_test[10:20, :]
    dif = imgs_y-imgs_x
    for j in range(10):
        trav = [imgs_x[j,:]+dif[j,:]*i*0.1 for i in range(11)]
        imgs = generate_image(model,torch.stack(trav))
        save_images_group(imgs,folder_path,'feature_trav',f'trav_{j}')





