from datasets.dataset_DIF import *
import pandas as pd
from torchvision.utils import save_image
import tqdm
from torch.cuda.amp import autocast,GradScaler
import os

def get_model(pretrained,map_location):
    weights = torch.load(pretrained,map_location=map_location)
    model = weights['model']
    return model
def get_dl(indicator,data_list,opt):
    train_set = ImageDatasetFromFile_DIF(indicator, data_list, opt.dataroot, input_height=None,
                                         crop_height=None, output_height=opt.output_height, is_mirror=False,
                                         is_gray=opt.cdim != 3)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=False,
                                                    num_workers=opt.workers)
    return train_data_loader

def dataloader_train_val_test(opt):
    data = pd.read_csv(opt.class_indicator_file)
    data = data.sample(frac=1,random_state=0)
    train_list_1 = data['file_name'].values.tolist()[:opt.trainsize]
    train_property_indicator_1 = data['class'].values.tolist()[:opt.trainsize]

    train_list = train_list_1[:(opt.trainsize-opt.valsize)]
    train_property_indicator = train_property_indicator_1[:(opt.trainsize-opt.valsize)]

    val_list = train_list_1[(opt.trainsize - opt.valsize):]
    val_property_indicator = train_property_indicator_1[(opt.trainsize - opt.valsize):]

    test_list = data['file_name'].values.tolist()[opt.trainsize:]
    test_property_indicator = data['class'].values.tolist()[opt.trainsize:]

    # swap out the train files

    assert len(train_list) > 0

    train_data_loader = get_dl(train_property_indicator, train_list, opt)
    val_data_loader = get_dl(val_property_indicator, val_list, opt)
    test_data_loader = get_dl(test_property_indicator, test_list, opt)

    return train_data_loader,val_data_loader,test_data_loader

def dataloader_train_test(opt):
    data = pd.read_csv(opt.class_indicator_file)
    data = data.sample(frac=1,random_state=0)
    train_list = data['file_name'].values.tolist()[:opt.trainsize]
    train_property_indicator = data['class'].values.tolist()[:opt.trainsize]

    test_list = data['file_name'].values.tolist()[opt.trainsize:]
    test_property_indicator = data['class'].values.tolist()[opt.trainsize:]

    # swap out the train files

    assert len(train_list) > 0

    train_data_loader = get_dl(train_property_indicator,train_list,opt)
    test_data_loader = get_dl(test_property_indicator,test_list,opt)


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

def save_images_group(images, dir, folder, file_name,nrow=8):
    if not os.path.exists(dir+folder):
        os.makedirs(dir+folder)
    save_image(images, dir + folder + f'/{file_name}.jpg',nrow)

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


def feature_traverse(latent,id,C,sign):
    ints=5
    trav = [C/ints*i-C for i in range(2*ints+1)]
    l = torch.zeros_like(latent)
    l[id]=1
    container = []
    for z in trav: #A-B
        container.append(latent-z*l*sign)
    return torch.stack(container,dim=0)


def feature_isolation(C, z_test, c_test, lasso_model, model,folder_path,alpha):
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
            sign = torch.sign(w.squeeze()[id])
            list_x.append(feature_traverse(imgs_x[j,:],id,C,sign))
            list_y.append(feature_traverse(imgs_y[j,:],id,C,sign))
        list_id_x.append(list_x)
        list_id_y.append(list_y)
    for i in range(5):
        for j in range(5):
            a = list_id_x[i][j]
            b = list_id_y[i][j]
            imgs_a = generate_image(model,a)
            save_images_group(imgs_a,folder_path,f'feature_isolate_A_{alpha}',f'isolate_feature_{i}_pic_{j}',nrow=11)
            imgs_b = generate_image(model,b)
            save_images_group(imgs_b,folder_path,f'feature_isolate_B_{alpha}',f'isolate_feature_{i}_pic_{j}',nrow=11)
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
        save_images_group(imgs,folder_path,'feature_trav',f'trav_{j}',nrow=11)





