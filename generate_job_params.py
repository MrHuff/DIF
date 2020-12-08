import os
import shutil
import pickle
job_base = {
    'prefix':"mnist_multi",
    'hdim': 16,
    'output_height':64,
    'channels':[64, 128, 256, 512],
    'm_plus':1000,
    'weight_rec': 1.0,
    'weight_kl':1.0,
    'weight_neg':0.5,
    'num_vae':10,
    'dataroot':'/data/ziz/rhu/data/mnist_full_64x64',
    'trainsize':60000,
    'test_iter':500,
    'save_iter':2,
    'start_epoch': 0,
    'batchSize':32,
    'nrow':8,
    'lr_e':0.0002,
    'lr_g':0.0002,
    'cuda':True,
    'nEpochs':25,
    'class_indicator_file':'/data/ziz/rhu/local_deploys/IntroVAE/mnist_full.csv',
    'fp_16':True,
    'J':0.25,
    'lambda_me':0,
    'C':10,
    'tanh_flag':True,
    'cdim':1,
    'kernel':'linear',
    'separation_objective':2,
    'apply_mask':True,
    'mask_KL':1.0
}
def save_obj(obj, name ,folder):
    with open(f'{folder}'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)

def mnist_jobs(job_dir):
    for i,lambda_me in enumerate([0,0.25,0.5]):
        job_base['lambda_me'] = lambda_me
        save_obj(job_base,f'mnist_job_{i}',job_dir)

def mnist_jobs_bench(job_dir):
    job_base['separation_objective']=1
    for i,lambda_me in enumerate([0.01,0.25,0.5]):
        job_base['lambda_me'] = lambda_me
        save_obj(job_base,f'mnist_job_bench_{i}',job_dir)

def mnist_jobs_VAE(job_dir):
    job_base['prefix']='mnist_multi_VAE'
    job_base['num_vae']=25
    job_base['nEpochs']=25
    for i,lambda_me in enumerate([0,0.01, 0.25, 0.5]):
        job_base['lambda_me'] = lambda_me
        save_obj(job_base, f'mnist_job_VAE_{i}', job_dir)


def celeba_jobs(job_dir):
    job_base['prefix']='faces_multi'
    job_base['hdim']=512
    job_base['output_height']=256
    job_base['channels']=[32, 64, 128, 256, 512, 512]
    job_base['dataroot'] = '/data/ziz/rhu/data/data256x256'
    job_base['trainsize']=29000
    job_base['test_iter']=1000
    job_base['save_iter']=10
    job_base['nEpochs']=200
    job_base['class_indicator_file']='/data/ziz/rhu/local_deploys/IntroVAE/celebA_hq_gender_multi.csv'
    job_base['cdim']=3
    for i,lambda_me in enumerate([0,0.5]):
        job_base['lambda_me'] = lambda_me
        save_obj(job_base, f'celebA_job_{i}', job_dir)

def celeba_jobs_VAE(job_dir):
    job_base['prefix']='faces_multi_VAE'
    job_base['hdim']=512
    job_base['output_height']=256
    job_base['channels']=[32, 64, 128, 256, 512, 512]
    job_base['dataroot'] = '/data/ziz/rhu/data/data256x256'
    job_base['trainsize']=29000
    job_base['test_iter']=1000
    job_base['save_iter']=10
    job_base['nEpochs']=100
    job_base['num_vae']=100
    job_base['class_indicator_file']='/data/ziz/rhu/local_deploys/IntroVAE/celebA_hq_gender_multi.csv'
    job_base['cdim']=3
    for i,lambda_me in enumerate([0,0.01, 0.25, 0.5]):
        job_base['lambda_me'] = lambda_me
        save_obj(job_base, f'celebA_job_VAE_{i}', job_dir)

def mpi3d_jobs():
    pass

if __name__ == '__main__':
    job_dir = 'intro_vae_jobs/'
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    else:
        shutil.rmtree(job_dir)
        os.makedirs(job_dir)

    celeba_jobs_VAE(job_dir)
    mnist_jobs_VAE(job_dir)
    celeba_jobs(job_dir)
    mnist_jobs(job_dir)