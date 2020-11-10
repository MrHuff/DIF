from post_processing_script import *
opt.batchSize = 1
opt.workers = 1
def load_model_and_data(opt):
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
    return model,dl_train,dl_test

def adversarial_experiment(opt,model,dl_test):
    c = torch.tensor(True)
    iterator = iter(dl_test)
    while c.item():
        batch_class_source, c  = next(iterator)
    batch_class_source = batch_class_source.to(base_gpu)
    c=torch.tensor(False)
    iterator = iter(dl_test)
    while not c.item():
        batch_class_target, c  = next(iterator)
    batch_class_target = batch_class_target.to(base_gpu)
    loss = torch.nn.MSELoss()
    lambas = [0]
    for lamb in lambas:
        d_img = torch.randn_like(batch_class_source, requires_grad=True)
        optimizer = torch.optim.Adam([d_img], lr=1e-2)
        for i in range(500):
            real_mu, real_logvar, z, rec = model(batch_class_source+d_img)
            l = loss(rec,batch_class_target) + torch.norm(d_img)*lamb
            print('Distortion loss: ', l.item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        d_img.requires_grad=False
        img_adversarial = batch_class_source+d_img
        with torch.no_grad():
            real_mu, real_logvar, z, rec_adversarial = model(img_adversarial)
        if not os.path.exists(opt.save_path+'adversarial_experiments'):
            os.makedirs(opt.save_path+'adversarial_experiments')
        save_image(rec_adversarial,opt.save_path+f'adversarial_experiments/rec_adversarial_{lamb}.jpg')
        save_image(batch_class_source,opt.save_path+'adversarial_experiments/source.jpg')
        save_image(batch_class_target,opt.save_path+'adversarial_experiments/target.jpg')
        save_image(d_img,opt.save_path+f'adversarial_experiments/noise_{lamb}.jpg')

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
    print(base_gpu)
    # zip([0, 1, 2, 3], [save_paths_mnist, save_paths_fashion, save_paths_faces, save_paths_covid],
    #     [model_paths_mnist, model_paths_fashion, model_paths_faces, model_paths_covid]):
    for c,a,b in zip([2],[save_paths_faces],[model_paths_faces]):
        opt.dataset_index = c  # 0 = mnist, 1 = fashion, 2 = celeb
        for i,el in enumerate(a):
            opt.save_path = el+'/'
            opt.load_path = opt.save_path+b[i]
            model, dl_train, dl_test = load_model_and_data(opt)
            adversarial_experiment(opt,model,dl_test)


