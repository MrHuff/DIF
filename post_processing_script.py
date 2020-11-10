from utils.model_utils import *
from utils.prototypes import *
from utils.umap import *
from utils.feature_isolation import *
from utils.FID import *
from models.DIF_net import *
from utils.loglikelihood import *
import GPUtil
import torch.backends.cudnn as cudnn
import pandas as pd
import shutil
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


opt = dotdict
opt.batchSize = 32
opt.J = 0.25
opt.use_flow_model = False
opt.cuda = True
opt.n_witness = 16
opt.cur_it = 123
opt.umap=False
opt.feature_isolation = True
opt.witness = False
opt.FID= False
opt.log_likelihood=False
opt.FID_fake = False
opt.FID_prototypes = False
opt.workers = 4
opt.C=10
dataroots_list = ["/homes/rhu/data/mnist_3_8_64x64/","/homes/rhu/data/fashion_256x256/","/homes/rhu/data/data256x256/","/homes/rhu/data/covid_dataset_256x256/"]
class_indicator_files_list = ["/homes/rhu/data/mnist_3_8.csv","/homes/rhu/data/fashion_price_class.csv","/homes/rhu/data/celebA_hq_gender.csv","/homes/rhu/data/covid_19_sick.csv"]
train_sizes = [13000,22000,29000,1900]
cdims = [1,3,3,1]
img_height=[64,256,256,256]
hdim_list=[16,512,512,512]


save_paths_faces = [
    'modelfacesHQv3_bs=32_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.25_kernel=linear_tanh=True_C=10.0_linearb=False',
                    'modelfacesHQv3_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.0_kernel=rbf_tanh=True_C=10.0_linearb=False',
                    'modelfacesHQv3_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=1.0_kernel=rbf_tanh=True_C=10.0_linearb=True'
]
model_paths_faces = [
    'model_epoch_130_iter_117882.pth',
    'model_epoch_130_iter_117888.pth',
    'model_epoch_130_iter_157146.pth'
]

save_paths_fashion = [
                    'modelfashion_bs=24_beta=1.0_KL=0.1_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.4_kernel=linear_tanh=True_C=10.0_linearb=False',
                      'modelfashion_bs=24_beta=1.0_KL=0.1_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.0_kernel=rbf_tanh=True_C=10.0_linearb=False',
                      'modelfashion_bs=24_beta=1.0_KL=0.1_KLneg=0.5_fd=3_m=1000.0_lambda_me=1.0_kernel=linear_tanh=True_C=10.0_linearb=True'
                      ]
model_paths_fashion = [
                        'model_epoch_180_iter_165060.pth',
                       'model_epoch_180_iter_165060.pth',
                        'model_epoch_180_iter_165060.pth'
                    ]
save_paths_mnist = ['modelmnist38_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.01_kernel=linear_tanh=True_C=10.0_linearb=False',
                    'modelmnist38_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=1.0_kernel=linear_tanh=True_C=10.0_linearb=True',
                    'modelmnist38_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.0_kernel=rbf_tanh=True_C=10.0_linearb=False']
model_paths_mnist = ['model_epoch_24_iter_9760.pth','model_epoch_24_iter_9761.pth','model_epoch_24_iter_9760.pth']

save_paths_covid = [
                    'modelcovid256_bs=24_beta=0.25_KL=1.0_KLneg=0.5_fd=3_m=150.0_lambda_me=0.15_kernel=linear_tanh=True_C=10.0_linearb=False_J=0.25',
                    'modelcovid256_bs=24_beta=0.25_KL=1.0_KLneg=0.5_fd=3_m=150.0_lambda_me=1.0_kernel=rbf_tanh=True_C=10.0_linearb=True_J=0.0',
                    'modelcovid256_bs=24_beta=0.25_KL=1.0_KLneg=0.5_fd=3_m=150.0_lambda_me=0.0_kernel=rbf_tanh=True_C=10.0_linearb=False_J=0.0',
                    ]
model_paths_covid = [
                    'model_epoch_150_iter_9222.pth',
                    'model_epoch_150_iter_12000.pth',
                    'model_epoch_150_iter_12000.pth',
                     ]

def run_post_process(opt,base_gpu,runs=1):
    opt.dataroot = dataroots_list[opt.dataset_index]
    opt.class_indicator_file = class_indicator_files_list[opt.dataset_index]
    opt.trainsize=train_sizes[opt.dataset_index]
    opt.cdim = cdims[opt.dataset_index]
    opt.hdim = hdim_list[opt.dataset_index]
    opt.output_height=img_height[opt.dataset_index]
    print(opt.dataroot)
    print(opt.class_indicator_file)
    map_location= f'cuda:{base_gpu}'
    model= get_model(opt.load_path,map_location)
    print(model)
    dl_train,dl_test = dataloader_train_test(opt)
    train_z,train_c = generate_all_latents(model=model,dataloader=dl_train)
    test_z,test_c = generate_all_latents(model=model,dataloader=dl_test)
    traverse(test_z, test_c, model, opt.save_path)
    if opt.umap:
        make_binary_class_umap_plot(train_z.cpu().float().numpy(),train_c.cpu().numpy(),opt.save_path,opt.cur_it,'umap_train')
        make_binary_class_umap_plot(test_z.cpu().float().numpy(),test_c.cpu().numpy(),opt.save_path,opt.cur_it,'umap_test')
    big_val = []
    for r in range(runs):
        val = []
        cols = []
        if opt.feature_isolation:
            alphas = [0,1e-3,1e-2,1e-1]
            for alp in alphas:

                lasso_model,test_auc = lasso_train(opt.save_path,train_z,train_c,test_z,test_c,alp,1e-3,100,bs_rate=1e-2)
                lasso_model.load_state_dict(torch.load(opt.save_path+f'lasso_latents_{alp}.pth',map_location=map_location))
                lasso_model.eval()
                weights_histogram(lasso_model,opt.save_path,alp)
                sparsity_level = get_feature_sparsity(lasso_model)
                cols.append(f'sparsity_level_{alp}')
                val.append(sparsity_level)
                with torch.no_grad():
                    preds = lasso_model(test_z)
                test_auc = auc_check(preds, test_c)
                cols.append(f'test_auc_{alp}')
                val.append(test_auc)
                feature_isolation(opt.C,test_z,test_c,lasso_model,model,opt.save_path,alp)
                #Fix direction of A,B to be which depend on the sign of the feature...
        if opt.witness:

            witness_obj, pval = training_loop_witnesses(opt.save_path,opt.hdim, opt.n_witness, train_z, train_c, test_z, test_c,init_type='gaussian_fit')

            witnesses_tensor = generate_image(model,witness_obj.T)
            cols.append('test_pval')
            val.append(pval)
            try:
                lasso_model = lasso_regression(in_dim=opt.hdim, o_dim=1).cuda()
                lasso_model.load_state_dict(torch.load(opt.save_path+'lasso_latents_0.pth',map_location))
                lasso_model.eval()
                preds = lasso_model(witness_obj.T)
                mask = preds>=0.5
                if os.path.exists(opt.save_path+'prototypes_A'):
                    shutil.rmtree(opt.save_path+'prototypes_A')
                save_images_individually(witnesses_tensor[~mask.squeeze(),:,:,:], opt.save_path, 'prototypes_A', 'prototype_A')
                if os.path.exists(opt.save_path+'prototypes_B'):
                    shutil.rmtree(opt.save_path+'prototypes_B')
                save_images_individually(witnesses_tensor[mask.squeeze(),:,:,:], opt.save_path, 'prototypes_B', 'prototype_B')
                save_images_individually(witnesses_tensor, opt.save_path, 'prototypes', 'prototype')

            except Exception as e:
                print(e)
                print("No classification model found, saving without classifying")
                save_images_individually(witnesses_tensor, opt.save_path, 'prototypes', 'prototype')

        if opt.FID:
            fake_tensor = get_fake_images(model,32)
            save_images_individually(fake_tensor, opt.save_path, 'fake_images', 'fake')
            datasets = ['mnist38','fashion','celebHQ','covid']
            for i,d in enumerate(datasets):
                if not os.path.isfile(f'./precomputed_fid/{d}/data.npy'):
                    if not os.path.exists(f'./precomputed_fid/{d}/'):
                        os.makedirs(f'./precomputed_fid/{d}/')
                    m1,s1= calculate_dataset_FID(dataroots_list[i],32,True,2048)
                    save_FID(m1,s1,f'./precomputed_fid/{d}')
            d = datasets[opt.dataset_index]
            m1,s1 = load_FID(f'./precomputed_fid/{d}')

            if opt.FID_fake:
                m_fake,s_fake = calculate_dataset_FID(opt.save_path+'fake_images/',32,True,2048)
                fid_fake = calculate_frechet_distance(m1,s1,m_fake,s_fake)
                cols.append('fake_FID')
                val.append(fid_fake)

            if opt.FID_prototypes:
                m_prototypes,s_prototypes = calculate_dataset_FID(opt.save_path+'prototypes/',32,True,2048)
                fid_prototypes = calculate_frechet_distance(m1, s1, m_prototypes, s_prototypes)
                cols.append('prototype_FID')
                val.append(fid_prototypes)

        if opt.log_likelihood:
            cols = cols + ['log-likelihood','ELBO','log-likelihood_A','log-likelihood_B','ELBO_A','ELBO_B']
            _loglikelihood_estimates,_elbo_estimates,_class = estimate_loglikelihoods(dl_test,model,500)
            print(_loglikelihood_estimates.shape)
            print(_elbo_estimates.shape)
            ll,elbo,ll_A,ll_B,elbo_A,elbo_B=calculate_metrics(_loglikelihood_estimates, _elbo_estimates, _class)
            val = val + [ll,elbo,ll_A,ll_B,elbo_A,elbo_B]
        big_val.append(val)
    df = pd.DataFrame(big_val,columns=cols)
    print(df)
    df.to_csv(opt.save_path+'results.csv')
    summary = df.describe()
    summary.to_csv(opt.save_path+'summary.csv')

if __name__ == '__main__':
    if opt.cuda:
        base_gpu_list = GPUtil.getAvailable(order='memory', limit=8)
        base_gpu = base_gpu_list[0]
        cudnn.benchmark = True
    elif torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        base_gpu = "cpu"
    else:
        base_gpu = "cpu"
    torch.cuda.set_device(base_gpu)

    for c,a,b in zip([0,1,2,3],[save_paths_mnist,save_paths_fashion,save_paths_faces,save_paths_covid],[model_paths_mnist,model_paths_fashion,model_paths_faces,model_paths_covid]):
        opt.dataset_index = c  # 0 = mnist, 1 = fashion, 2 = celeb
        for i,el in enumerate(a):
            opt.save_path = el+'/'
            opt.load_path = opt.save_path+b[i]
            run_post_process(opt,base_gpu,runs=3)












