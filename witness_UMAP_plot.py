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

def generate_umaps_witness(opt,epochs_list = [0]):
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
        X = train_z[~train_c, :]
        Y = train_z[train_c, :]
        tr_nx = round(X.shape[0] * 0.9)
        tr_ny = round(Y.shape[0] * 0.9)
        witness_obj = witness_generation(opt.hdim, opt.n_witness, X[:tr_nx, :], Y[:tr_ny, :]).cuda()
        witness_obj.load_state_dict(torch.load(opt.save_path + 'witness_object.pth', map_location))
        witness_obj.eval()
        plot_witnesses(train_z.cpu().float().numpy(), train_c.cpu().numpy(), opt.save_path, opt.cur_it,'umap_train',witness_obj.T.detach().cpu().numpy())
        plot_witnesses(test_z.cpu().float().numpy(), test_c.cpu().numpy(), opt.save_path, opt.cur_it, 'umap_test',witness_obj.T.detach().cpu().numpy())

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
    epoch_list = [150]
    for c,a in zip([3],[save_paths_covid]):
        opt.dataset_index = c  # 0 = mnist, 1 = fashion, 2 = celeb
        for i, el in enumerate(a):
            opt.save_path = el+'/'
            generate_umaps_witness(opt,[epoch_list[0]])

