from models.networks_v2 import Classifier
from datasets.dataset import *
import torch
from utils.loglikelihood import *
import os
from os import listdir
import GPUtil
import torch.backends.cudnn as cudnn
import pandas as pd
import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



opt = dotdict
opt.dataset_index = 2 #0 = mnist, 1 = fashion, 2 = celeb
opt.batchSize = 32
opt.J = 0.25
opt.C = 10
opt.tanh_flag = True
opt.use_flow_model = False
opt.cuda = True
save_paths = ['mnist_classify/','fashion_classify/','celeb_classify/']
channels = [[32, 64, 128, 256],[32, 64, 128, 256, 512, 512],[32, 64, 128, 256, 512, 512]]
opt.workers = 4
cdims = [1,3,3]
dataroots_list = ["/homes/rhu/data/mnist_3_8_64x64/","/homes/rhu/data/fashion_256x256/","/homes/rhu/data/data256x256/"]
class_indicator_files_list = ["/homes/rhu/data/mnist_3_8.csv","/homes/rhu/data/fashion_price_class.csv","/homes/rhu/data/celebA_hq_gender.csv"]
train_sizes = [13000,22000,29000]
image_size = [64,256,256]
epochs = 100
if __name__ == '__main__':
    if opt.cuda:
        base_gpu_list = GPUtil.getAvailable(order='memory', limit=8)
        if 5 in base_gpu_list:
            base_gpu_list.remove(5)
        base_gpu = base_gpu_list[0]
        cudnn.benchmark = True
    elif torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    torch.cuda.set_device(base_gpu)
    dl_train, dl_val, dl_test = dataloader_train_val_test(opt)
    opt.channels = channels[opt.dataset_index]
    opt.save_path = save_paths[opt.dataset_index]
    model = Classifier(cdims[opt.dataset_index],opt.channels,image_size[opt.dataset_index]).cuda()
    for i in tqdm.trange(epochs):
        pass

        #trainloop
        #validationloop - auc
        #testloop - auc

    #save model and auc performance.
