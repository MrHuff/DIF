from utils.model_utils import *
from utils.prototypes import *
from utils.umap import *
from utils.feature_isolation import *
from utils.FID import *
from models.DIF_net import *
import GPUtil
import torch.backends.cudnn as cudnn
import pandas as pd

opt = dotdict
opt.class_indicator_file = "/homes/rhu/data/celebA_hq_gender.csv"
opt.trainsize = 29000
opt.dataroot = "/homes/rhu/data/data256x256"
opt.output_height = 256
opt.batchSize = 32
opt.J = 0.25
opt.C = 10
opt.tanh_flag = True
opt.channels = [32, 64, 128, 256, 512, 512]
opt.use_flow_model = False
opt.cuda = True
opt.trainsize = 29000
opt.save_path = 'model_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.0_kernel=rbf_tanh=False_C=100.0/'
opt.load_path = opt.save_path+'model_epoch_160_iter_145078.pth'
opt.hdim = 512
opt.n_witness = 16
opt.cur_it = 145083
opt.umap=False
opt.feature_isolation = False
opt.witness = False
opt.FID= True
opt.workers = 4
opt.flow_depth = 4
opt.cdim = 3
if __name__ == '__main__':

    if opt.cuda:
        base_gpu_list = GPUtil.getAvailable(order='memory', limit=2)
        if 5 in base_gpu_list:
            base_gpu_list.remove(5)
        base_gpu = base_gpu_list[0]
        cudnn.benchmark = True
    elif torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    torch.cuda.set_device(base_gpu)
    if opt.use_flow_model:
        model = DIF_net_flow(flow_C=opt.C,
                    flow_depth=opt.flow_depth,
                    tanh_flag=opt.tanh_flag,
                    cdim=opt.cdim,
                    hdim=opt.hdim,
                    channels=opt.channels,
                    image_size=opt.output_height).cuda()
    else:
        model = DIF_net(
            flow_C=opt.C,
            flow_depth=opt.flow_depth,
            tanh_flag=opt.tanh_flag,
            cdim=opt.cdim,
            hdim=opt.hdim,
            channels=opt.channels,
            image_size=opt.output_height).cuda()
    load_model(model,opt.load_path)
    model.eval()
    dl_train,dl_test = dataloader_train_test(opt)
    train_z,train_c = generate_all_latents(model=model,dataloader=dl_train)
    test_z,test_c = generate_all_latents(model=model,dataloader=dl_test)
    if opt.umap:
        make_binary_class_umap_plot(train_z.cpu().numpy(),train_c.cpu().numpy(),opt.save_path,opt.cur_it,'umap_train')
        make_binary_class_umap_plot(test_z.cpu().numpy(),test_c.cpu().numpy(),opt.save_path,opt.cur_it,'umap_test')
    if opt.feature_isolation:
        lasso_model,test_auc = lasso_train(train_z,train_c,test_z,test_c,0.1,1e-2,100,bs_rate=1e-2)
    if opt.witness:
        witness_obj, pval = training_loop_witnesses(opt.hdim, opt.n_witness, train_z, train_c, test_z, test_c)
    if opt.FID:
        datasets = ['mnist38','fashion','celebHQ']
        d_paths = ["/homes/rhu/data/mnist_3_8_64x64","/homes/rhu/data/fashion_256x256","/homes/rhu/data/data256x256"]
        for i,d in enumerate(datasets):
            if not os.path.isfile(f'./precomputed_fid/{d}/data.npy'):
                if not os.path.exists(f'./precomputed_fid/{d}/'):
                    os.makedirs(f'./precomputed_fid/{d}/')
                m1,s1= calculate_dataset_FID(d_paths[i],32,True,2048)
                save_FID(m1,s1,f'./precomputed_fid/{d}')









