from models.networks_v2 import Classifier
from utils.loglikelihood import *
from utils.feature_isolation import auc_check
import GPUtil
import torch.backends.cudnn as cudnn
import pandas as pd
import tqdm
from torch.cuda.amp import autocast,GradScaler
from models.ME_objectives import stableBCEwithlogits
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

opt = dotdict
opt.print_interval = 50
opt.dataset_index = 2 #0 = mnist, 1 = fashion, 2 = celeb
opt.batchSize = 32
opt.J = 0.25
opt.C = 10
opt.tanh_flag = True
opt.use_flow_model = False
opt.cuda = True
save_paths = ['mnist_classify/','fashion_classify/','celeb_classify/']
channels = [[32, 64, 128, 256],[32, 64, 128, 256, 512, 512],[32, 64, 128, 256, 512, 512]]
opt.workers = 8
cdims = [1,3,3]
dataroots_list = ["/homes/rhu/data/mnist_3_8_64x64/","/homes/rhu/data/fashion_256x256/","/homes/rhu/data/data256x256/"]
class_indicator_files_list = ["/homes/rhu/data/mnist_3_8.csv","/homes/rhu/data/fashion_price_class.csv","/homes/rhu/data/celebA_hq_gender.csv"]
train_sizes = [13000,22000,29000]
val_sizes = [500,500,500]
image_size = [64,256,256]
epochs = 100

def val_loop(dl,model):
    c_cat = []
    pred_cat = []
    for iteration, (batch, c) in enumerate(tqdm.tqdm(dl)):
        with torch.no_grad():
            batch = batch.cuda()
            c = c.cuda()
            preds = model(batch)
            pred_cat.append(preds)
            c_cat.append(c)
    pred_cat = torch.cat(pred_cat,dim=0)
    c_cat = torch.cat(c_cat,dim=0)
    return auc_check(pred_cat,c_cat)

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

    opt.channels = channels[opt.dataset_index]
    opt.save_path = save_paths[opt.dataset_index]
    opt.class_indicator_file = class_indicator_files_list[opt.dataset_index]
    opt.dataroot = dataroots_list[opt.dataset_index]
    opt.trainsize = train_sizes[opt.dataset_index]
    opt.valsize = val_sizes[opt.dataset_index]
    opt.output_height = image_size[opt.dataset_index]
    opt.cdim = cdims[opt.dataset_index]
    dl_train, dl_val, dl_test = dataloader_train_val_test(opt)
    model = Classifier(cdims[opt.dataset_index],opt.channels,image_size[opt.dataset_index]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    pos_weight = torch.tensor((len(dl_train.dataset.property_indicator)-sum(dl_train.dataset.property_indicator))/sum(dl_train.dataset.property_indicator)).cuda()
    objective = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=0.5)
    for i in tqdm.trange(epochs):
        for iteration, (batch, c) in enumerate(tqdm.tqdm(dl_train)):
            with autocast():
                batch = batch.cuda()
                c = c.cuda()
                pred = model(batch)
                loss = objective(pred,c.float().squeeze())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # .step()
            scaler.update()
            if iteration%opt.print_interval==0:
                print(f'Iteration {i} trainloss={loss.item()}')
        val_auc = val_loop(dl_val,model)
        lrs.step(-val_auc)
        test_auc = val_loop(dl_test,model)
        print(f'val auc {i}:', val_auc)
        print(f'val auc {i}:', test_auc)

    df = pd.DataFrame([val_auc,test_auc],columns=['val auc','test auc'])
    print(df)
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    df.to_csv(opt.save_path+'performance.csv')
