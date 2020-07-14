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
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
opt = dotdict
opt.dataset_index = 2 #0 = mnist, 1 = fashion, 2 = celeb
opt.output_height = 256
opt.batchSize = 32
opt.J = 0.25
opt.C = 10
opt.tanh_flag = True
opt.channels = [32, 64, 128, 256, 512, 512]
opt.use_flow_model = False
opt.cuda = True
opt.save_path = 'model_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.0_kernel=rbf_tanh=False_C=100.0/'#'model_fashion_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.2_kernel=linear_tanh=True_C=10.0/'#'model_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.0_kernel=rbf_tanh=False_C=100.0/'
opt.load_path = opt.save_path+'model_epoch_160_iter_145078.pth'#'model_epoch_180_iter_123840.pth' #'model_epoch_160_iter_145078.pth'
opt.hdim = 512
opt.n_witness = 16
opt.cur_it = 145078
opt.umap=False
opt.feature_isolation = False
opt.witness = False
opt.FID= False
opt.log_likelihood=True
opt.workers = 4
opt.flow_depth = 4
opt.cdim = 3
dataroots_list = ["/homes/rhu/data/mnist_3_8_64x64/","/homes/rhu/data/fashion_256x256/","/homes/rhu/data/data256x256/"]
class_indicator_files_list = ["/homes/rhu/data/mnist_3_8.csv","/homes/rhu/data/fashion_price_class.csv","/homes/rhu/data/celebA_hq_gender.csv"]
train_sizes = [13000,22000,29000]
opt.FID_fake = True
opt.FID_prototypes = True

if __name__ == '__main__':
    opt.dataroot = dataroots_list[opt.dataset_index]
    opt.class_indicator_file = class_indicator_files_list[opt.dataset_index]
    opt.trainsize=train_sizes[opt.dataset_index]
    print(opt.dataroot)
    print(opt.class_indicator_file)
    cols = []
    val = []
    if opt.cuda:
        base_gpu_list = GPUtil.getAvailable(order='memory', limit=8)
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
    map_location= f'cuda:{base_gpu}'
    load_model(model,opt.load_path,map_location)
    model.eval()
    dl_train,dl_test = dataloader_train_test(opt)
    train_z,train_c = generate_all_latents(model=model,dataloader=dl_train)
    test_z,test_c = generate_all_latents(model=model,dataloader=dl_test)
    if opt.umap:
        make_binary_class_umap_plot(train_z.cpu().numpy(),train_c.cpu().numpy(),opt.save_path,opt.cur_it,'umap_train')
        make_binary_class_umap_plot(test_z.cpu().numpy(),test_c.cpu().numpy(),opt.save_path,opt.cur_it,'umap_test')
    if opt.feature_isolation:
        try:
            lasso_model = lasso_regression(in_dim=opt.hdim, o_dim=1).cuda()
            lasso_model.load_state_dict(torch.load(opt.save_path+'lasso_latents.pth',map_location=map_location))
            lasso_model.eval()
            test_auc = auc_check(lasso_model,test_z,test_c)
        except Exception as e:
            print(e)
            print("No latent regression model exists, training a new one!")
            lasso_model,test_auc = lasso_train(train_z,train_c,test_z,test_c,0.1,1e-2,100,bs_rate=1e-2)
            torch.save(lasso_model.state_dict(), opt.save_path + 'lasso_latents.pth')

        cols.append('test_auc')
        val.append(test_auc)
        feature_isolation(opt.C,test_z,test_c,lasso_model,model,opt.save_path)
        traverse(test_z,test_c,model,opt.save_path)
    if opt.witness:
        #add load clause
        try:
            X = train_z[~train_c, :]
            Y = train_z[train_c, :]
            tr_nx = round(X.shape[0] * 0.9)
            tr_ny = round(Y.shape[0] * 0.9)
            witness_obj = witness_generation(opt.hdim, opt.n_witness,X[:tr_nx,:], Y[:tr_ny,:]).cuda()
            witness_obj.load_state_dict(torch.load(opt.save_path+'witness_object.pth',map_location))
            witness_obj.eval()
            tst_stat_test = witness_obj(test_z[~test_c,:], test_z[test_c,:])
            pval = witness_obj.get_pval_test(tst_stat_test.item())
        except Exception as e:
            print(e)
            print("No witness model exists, training a new one!")
            witness_obj, pval = training_loop_witnesses(opt.hdim, opt.n_witness, train_z, train_c, test_z, test_c)
            torch.save(witness_obj.state_dict(),opt.save_path+'witness_object.pth')

        witnesses_tensor = generate_image(model,witness_obj.T)
        cols.append('test_pval')
        val.append(pval)
        try:
            lasso_model = lasso_regression(in_dim=opt.hdim, o_dim=1).cuda()
            lasso_model.load_state_dict(torch.load(opt.save_path+'lasso_latents.pth',map_location))
            lasso_model.eval()
            preds = lasso_model(witness_obj.T)
            mask = preds>=0.5
            save_images_individually(witnesses_tensor[~mask.squeeze(),:,:,:], opt.save_path, 'prototypes_A', 'prototype_A')
            save_images_individually(witnesses_tensor[mask.squeeze(),:,:,:], opt.save_path, 'prototypes_B', 'prototype_B')
            save_images_individually(witnesses_tensor, opt.save_path, 'prototypes', 'prototype')

        except Exception as e:
            print(e)
            print("No classification model found, saving without classifying")
            save_images_individually(witnesses_tensor, opt.save_path, 'prototypes', 'prototype')

    if opt.FID:
        fake_tensor = get_fake_images(model,32)
        save_images_individually(fake_tensor, opt.save_path, 'fake_images', 'fake')
        datasets = ['mnist38','fashion','celebHQ']
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
        _loglikelihood_estimates,_elbo_estimates,_class = estimate_loglikelihoods(dl_test,model,50)
        print(_loglikelihood_estimates.shape)
        print(_elbo_estimates.shape)
        ll,elbo,ll_A,ll_B,elbo_A,elbo_B=calculate_metrics(_loglikelihood_estimates, _elbo_estimates, _class)
        val = val + [ll,elbo,ll_A,ll_B,elbo_A,elbo_B]

    df = pd.DataFrame([val],columns=cols)
    print(df)
    df.to_csv(opt.save_path+'results.csv')
















