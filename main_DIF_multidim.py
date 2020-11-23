from __future__ import print_function
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from datasets.dataset_DIF import *
import time
import random
import torchvision.utils as vutils
from models.networks import *
from models.DIF_net import *
from models.ME_objectives import *
from torch.cuda.amp import autocast,GradScaler
import pandas as pd
import GPUtil
from main import parser,record_image,record_scalar,str_to_list,load_model,save_checkpoint


parser.add_argument('--class_indicator_file', default="/home/file.csv", type=str, help='class indicator csv file')
parser.add_argument('--fp_16', action='store_true', help='enables fp_16')
parser.add_argument('--tanh_flag', action='store_true', help='enables tanh')
parser.add_argument("--C", type=float, default=100.0, help="Default=100.0")
parser.add_argument("--J", type=float, default=0.25, help="Default=0.25")
parser.add_argument("--kernel", default="rbf", type=str, help="kernel choice")
parser.add_argument("--lambda_me", type=float, default=1.0, help="Default=0.25")
parser.add_argument('--umap', action='store_true', help='visualizes umap')
parser.add_argument("--KL_G", type=float, default=0.25, help="KL_G")
parser.add_argument("--prefix", default="", type=str, help="dataset")
parser.add_argument('--separation_objective',type=int,default=1, help='linear bench')
parser.add_argument('--cdim', type=int, default=3, help='cdim')
parser.add_argument('--apply_mask', action='store_true', help='isolate features to specific dimensions')
parser.add_argument('--mask_KL', type=float, default=1.0, help="KL_mask")

def main():
    print(torch.__version__)
    # torch.autograd.set_detect_anomaly(True)
    global opt, model
    opt = parser.parse_args()
    print(opt)
    param_suffix = f"{opt.prefix}_bs={opt.batchSize}_beta={opt.weight_rec}_KL={opt.weight_kl}_KLneg={opt.weight_neg}_m={opt.m_plus}_lambda_me={opt.lambda_me}_kernel={opt.kernel}_tanh={opt.tanh_flag}_C={opt.C}_obj={opt.separation_objective}_J={opt.J}_mask={opt.apply_mask}_mask_KL={opt.mask_KL}"
    opt.outf = f'results{param_suffix}/'
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        base_gpu_list= GPUtil.getAvailable(order='memory',limit=2)
        if 5 in base_gpu_list:
            base_gpu_list.remove(5)
        base_gpu = base_gpu_list[0]
        cudnn.benchmark = True
    elif torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    #--------------build models -------------------------
    model = DIF_netv2(flow_C=opt.C,
                    tanh_flag=opt.tanh_flag,
                    cdim=opt.cdim,
                    hdim=opt.hdim,
                    channels=str_to_list(opt.channels),
                    image_size=opt.output_height).cuda(base_gpu)

    if opt.pretrained:
        load_model(model, opt.pretrained,map_location=base_gpu)
    print(model)
            
    optimizerE = optim.Adam(model.encoder.parameters(), lr=opt.lr_e)
    optimizerG = optim.Adam(model.decoder.parameters(), lr=opt.lr_g)
    if opt.fp_16:
        scaler = GradScaler()

    #-----------------load dataset--------------------------
    train_data = pd.read_csv(opt.class_indicator_file)
    train_data = train_data.sample(frac=1)#Shuffle your data!
    train_list = train_data['file_name'].values.tolist()[:opt.trainsize]
    property_indicator = train_data['class'].values.tolist()[:opt.trainsize]

    assert len(train_list) > 0
    
    train_set = ImageDatasetFromFile_DIF_multi(property_indicator,train_list, opt.dataroot, input_height=None, crop_height=None, output_height=opt.output_height, is_mirror=True,is_gray=opt.cdim!=3)
    nr_of_classes,label_counts = train_set.get_label_data()
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    if opt.apply_mask:
        mask = torch.Tensor([True]*nr_of_classes + [False]*(opt.hdim-nr_of_classes)).cuda(base_gpu)
    else:
        mask=None

    if opt.lambda_me!=0:
        if opt.separation_objective==2:
            me_obj = linear_benchmark(d=opt.hdim).cuda(base_gpu)
        elif opt.separation_objective==1:
            me_obj = NFSIC(J=opt.J,kernel_type=opt.kernel).cuda(base_gpu)

    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=opt.outf)
    
    start_time = time.time()
    cur_iter = 0

    def mask_KL(mean,logvar):
        loss_kl = opt.mask_KL * model.kl_loss(mean[:, mask], logvar[:,mask]).mean() + model.kl_loss(
            mean[:, ~mask], logvar[:, ~mask]).mean()
        return loss_kl.mean()

    def train_vae(epoch, iteration, batch,Y,cur_iter):
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        real= Variable(batch).cuda(base_gpu) 
                
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time.time()-start_time)
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
        #=========== Update E ================

        def VAE_forward():
            # real_mu, real_logvar, z_real, rec, flow_log_det_real, xi_real = model(real)
            real_mu, real_logvar, z_real, rec = model(real)
            loss_rec = model.reconstruction_loss(rec, real, True)
            if opt.apply_mask:
                loss_kl = mask_KL(mean=real_mu,logvar=real_logvar)
            else:
                loss_kl = model.kl_loss(real_mu, real_logvar).mean() #- flow_log_det_real.mean()

            if opt.lambda_me==0:
                T = torch.zeros_like(loss_rec)
            else:
                if opt.apply_mask:
                    T = me_obj(z_real[:,mask],Y)*opt.lambda_me
                else:
                    T = me_obj(z_real,Y)*opt.lambda_me

            loss = loss_rec + loss_kl - T
            return loss,loss_rec,loss_kl,rec,T

        if opt.fp_16:
            with autocast():
                loss,loss_rec,loss_kl,rec,T = VAE_forward()
            optimizerG.zero_grad()
            optimizerE.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizerE)  # .step()
            scaler.step(optimizerG)  # .step()
            scaler.update()
        else:
            loss,loss_rec,loss_kl,rec,T = VAE_forward()
            loss.backward()
            optimizerE.step()
            optimizerG.step()
     
        info += 'Rec: {:.4f}, KL: {:.4f}, ME: {:.4f} '.format(loss_rec.item(), loss_kl.item(),T.item())
        print(info)
        
        if cur_iter % opt.test_iter is 0:  
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                if cur_iter % 1000 == 0:
                    record_image(writer, [real, rec], cur_iter)   
            else:
                vutils.save_image(torch.cat([real, rec], dim=0).data.cpu(), '{}/vae_image_{}.jpg'.format(opt.outf, cur_iter),nrow=opt.nrow)
    
    def train(epoch, iteration, batch,Y, cur_iter):
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)
            
        batch_size = batch.size(0)
        Y = Y.cuda(base_gpu)
        noise = torch.randn(batch_size, opt.hdim).cuda(base_gpu)
        # noise_logvar = torch.zeros_like(noise)
        real= batch.cuda(base_gpu)
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time.time()-start_time)
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'

        #Problem is flow is trained with competing objectives on the same entity? Still unstable training!
        # Tune parameters?! Fake part is giving me a hard time...

        def update_E():

            fake = model.sample(noise)
            real_mu, real_logvar, z, rec = model(real)
            rec_mu, rec_logvar = model.encode(rec.detach())
            fake_mu, fake_logvar = model.encode(fake.detach())
            loss_rec =  model.reconstruction_loss(rec, real, True)

            if opt.apply_mask:
                lossE_real_kl =mask_KL(mean=real_mu,logvar=real_logvar)
                lossE_rec_kl = mask_KL(mean=rec_mu,logvar=rec_logvar)
                lossE_fake_kl = mask_KL(mean=fake_mu,logvar=fake_logvar)
            else:
                lossE_real_kl = model.kl_loss(real_mu, real_logvar).mean()
                lossE_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
                lossE_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()

            loss_margin = lossE_real_kl +(torch.relu(opt.m_plus-lossE_rec_kl)+torch.relu(opt.m_plus-lossE_fake_kl)) * 0.5 * opt.weight_neg
            if opt.lambda_me==0:
                T = torch.zeros_like(loss_rec)
            else:
                if opt.apply_mask:
                    T = me_obj(z[:,mask],Y)*opt.lambda_me
                else:
                    T = me_obj(z,Y)*opt.lambda_me
            # Also, ok might want to add more parametrization of hyper parameters.
            #weight neg should control adversarial objective. Want fakes and (reconstructions?!) to deviate from prior, want reals to be close to prior.
            #Don't know why reconstructions should be adversarial... Might want to rebalance
            lossE = loss_rec  * opt.weight_rec + loss_margin * opt.weight_kl-T
            return lossE,rec,fake,loss_rec,lossE_real_kl,\
                   lossE_rec_kl,lossE_fake_kl,real_logvar,rec_logvar,loss_margin,T

        #=========== Update E ================
        if opt.fp_16:
            with autocast():
                lossE,rec,fake,loss_rec,lossE_real_kl,\
                lossE_rec_kl,lossE_fake_kl,\
                real_logvar,rec_logvar,loss_margin,T_loss= update_E()
            optimizerG.zero_grad()
            optimizerE.zero_grad()
            scaler.scale(lossE).backward(retain_graph=True)
        else:
            lossE,rec,fake,loss_rec,lossE_real_kl,lossE_rec_kl,\
            lossE_fake_kl,real_logvar\
                ,rec_logvar,loss_margin,T_loss = update_E()
            optimizerG.zero_grad()
            optimizerE.zero_grad()
            lossE.backward(retain_graph=True)

        for m in model.encoder.parameters():
            m.requires_grad=False
        #========= Update G ==================
        def update_G():
            rec_mu, rec_logvar = model.encode(rec)
            fake_mu, fake_logvar = model.encode(fake)
            if opt.apply_mask:
                lossG_rec_kl = mask_KL(rec_mu, rec_logvar)
                lossG_fake_kl = mask_KL(fake_mu, fake_logvar)
            else:
                lossG_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
                lossG_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()
            lossG = (lossG_rec_kl + lossG_fake_kl) * 0.5 * opt.weight_kl
            return lossG,lossG_rec_kl,lossG_fake_kl

        if opt.fp_16:
            with autocast():
                lossG,lossG_rec_kl,lossG_fake_kl = update_G()
            scaler.scale(lossG).backward()
            scaler.step(optimizerE)  # .step()
            scaler.step(optimizerG)  # .step()
            scaler.update()
        else:
            lossG,lossG_rec_kl,lossG_fake_kl = update_G()
            lossG.backward()
            optimizerE.step()
            optimizerG.step()
        for m in model.encoder.parameters():
            m.requires_grad = True
        #. The key is to hold the regularization term LREG in Eq. (11) and Eq. (12) below the margin value m for most of the time
        info += 'Rec: {:.4f}, '.format(loss_rec.item()*opt.weight_rec)
        info += 'Margin loss: {:.4f}, '.format(opt.weight_kl*loss_margin.item())
        info += 'Total loss E: {:.4f}, '.format(lossE.item())
        info += 'Total loss G: {:.4f}, '.format(lossG.item())
        info += 'Kl_E: {:.4f}, {:.4f}, {:.4f}, '.format(lossE_real_kl.item(),
                                lossE_rec_kl.item(), lossE_fake_kl.item())
        info += 'Kl_G: {:.4f}, {:.4f}, '.format(lossG_rec_kl.item(), lossG_fake_kl.item())
        info += 'ME_flow: {:.4f}'.format(T_loss.item())

        print(info)
        
        if cur_iter % opt.test_iter is 0:            
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                if cur_iter % 1000 == 0:
                    record_image(writer, [real, rec, fake], cur_iter)   
            else:
                vutils.save_image(torch.cat([real, rec, fake], dim=0).data.cpu(), '{}/image_{}.jpg'.format(opt.outf, cur_iter),nrow=opt.nrow)

    #----------------Train by epochs--------------------------
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):  
        #save models
        if epoch % opt.save_iter == 0:
            save_checkpoint(model, epoch, cur_iter, '', folder_name=f"model{param_suffix}")

        model.train()
        for iteration, (batch,Y) in enumerate(train_data_loader, 0):
            #--------------train------------
            if epoch < opt.num_vae:
                train_vae(epoch, iteration, batch,Y, cur_iter)
            else:
                train(epoch, iteration, batch,Y, cur_iter)

            cur_iter += 1



if __name__ == "__main__":
    main()    