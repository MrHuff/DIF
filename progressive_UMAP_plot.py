import torch
from post_processing_script import *
from utils.umap import *
import os

#model_epoch_130_iter_117888.pth
def get_file_list(save_path,epoch_list):
    files = []
    for f in os.listdir(save_path):
        if f.endswith(".pth"):
            for u in epoch_list:
                exist_string = f'model_epoch_{u}_iter'
                if exist_string in f:
                    files.append(f)
    print(files)
    return files

def generate_umaps(opt,epochs_list = [0]):
    opt.dataroot = dataroots_list[opt.dataset_index]
    opt.class_indicator_file = class_indicator_files_list[opt.dataset_index]
    opt.trainsize = train_sizes[opt.dataset_index]
    opt.cdim = cdims[opt.dataset_index]
    opt.hdim = hdim_list[opt.dataset_index]
    opt.output_height = img_height[opt.dataset_index]
    print(opt.dataroot)
    print(opt.class_indicator_file)
    map_location = f'cuda:{base_gpu}'
    dl_train,dl_test = dataloader_train_test(opt)
    files = get_file_list(opt.save_path,epochs_list)
    for f in files:
        opt.load_path = opt.save_path+f
        model = get_model(opt.load_path, map_location)
        opt.cur_it= f
        train_z, train_c = generate_all_latents(model=model, dataloader=dl_train)
        test_z, test_c = generate_all_latents(model=model, dataloader=dl_test)
        make_binary_class_umap_plot(train_z.cpu().float().numpy(), train_c.cpu().numpy(), opt.save_path, opt.cur_it,'umap_train')
        make_binary_class_umap_plot(test_z.cpu().float().numpy(), test_c.cpu().numpy(), opt.save_path, opt.cur_it, 'umap_test')

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
    max_epoch = 180 #Adjust this according to model
    for c,a in zip([3],[save_paths_covid]):
        opt.dataset_index = c  # 0 = mnist, 1 = fashion, 2 = celeb
        for i, el in enumerate(a):
            print(el)
            opt.save_path = el+'/'
            generate_umaps(opt,[0,60,120,130,140,150])

