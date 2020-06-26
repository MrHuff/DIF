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
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
import pandas as pd
import GPUtil
from utils.umap import make_binary_class_umap_plot
from main_DIF import parser,record_image,record_scalar,str_to_list,save_checkpoint,load_model,subset_latents
from itertools import chain
#Did stabilize training... but why is the ME-stat negative?!
def main():
    print(torch.__version__)
    # torch.autograd.set_detect_anomaly(True)
    global opt, model
    opt = parser.parse_args()
    print(opt)
    param_suffix = f"_{opt.prefix}_flow_beta={opt.weight_rec}_KL={opt.weight_kl}_KLneg={opt.weight_neg}_fd={opt.flow_depth}_m={opt.m_plus}_lambda_me={opt.lambda_me}_kernel={opt.kernel}_tanh={opt.tanh_flag}_C={opt.C}"
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
    model = DIF_net_flow(flow_C=opt.C,
                    flow_depth=opt.flow_depth,
                    tanh_flag=opt.tanh_flag,
                    cdim=opt.cdim,
                    hdim=opt.hdim,
                    channels=str_to_list(opt.channels),
                    image_size=opt.output_height).cuda(base_gpu)

    # model = IntroVAE(cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height).cuda()

    if opt.pretrained:
        load_model(model, opt.pretrained)
    print(model)
            
    optimizerE = optim.Adam(chain(model.encoder.parameters(),model.flow.parameters()), lr=opt.lr_e)
    optimizerG = optim.Adam(model.decoder.parameters(), lr=opt.lr_g)
    if opt.fp_16:
        scaler = GradScaler()

    #-----------------load dataset--------------------------
    train_data = pd.read_csv(opt.class_indicator_file)
    train_list = train_data['file_name'].values.tolist()[:opt.trainsize]
    property_indicator = train_data['class'].values.tolist()[:opt.trainsize]
    #swap out the train files

    assert len(train_list) > 0
    
    train_set = ImageDatasetFromFile_DIF(property_indicator,train_list, opt.dataroot, input_height=None, crop_height=None, output_height=opt.output_height, is_mirror=True)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    if opt.linear_benchmark:
        me_obj = linear_benchmark(d=opt.hdim)
    else:
        me_obj = MEstat(J=opt.J,
                        test_nx=len(train_set.property_indicator) - sum(train_set.property_indicator),
                        test_ny=sum(train_set.property_indicator),
                        asymp_n=opt.asymp_n,
                        kernel_type=opt.kernel).cuda(base_gpu)
    min_features = 1./opt.J
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=opt.outf)
    
    start_time = time.time()
            
    cur_iter = 0
    
    def train_vae(epoch, iteration, batch,c,cur_iter):
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        real= Variable(batch).cuda(base_gpu) 
                
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time.time()-start_time)
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
        #=========== Update E ================

        def VAE_forward():
            real_mu, real_logvar, z_real, rec, flow_log_det_real, xi_real = model(real)
            loss_rec = model.reconstruction_loss(rec, real, True)
            loss_kl = model.kl_loss(real_mu, real_logvar).mean() - flow_log_det_real.mean()
            if opt.lambda_me==0:
                T = torch.zeros_like(loss_rec)
            else:
                z = model.flow_forward_only(xi_real.detach(),real_logvar.detach())
                T = me_obj(z, c)*opt.lambda_me
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
    
    def train(epoch, iteration, batch,c, cur_iter):
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)
            
        batch_size = batch.size(0)
        c = c.cuda(base_gpu)
        noise = torch.randn(batch_size, opt.hdim).cuda(base_gpu)
        noise_logvar = torch.zeros_like(noise)
        real= batch.cuda(base_gpu)
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time.time()-start_time)
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'

        #Problem is flow is trained with competing objectives on the same entity? Still unstable training!
        # Tune parameters?! Fake part is giving me a hard time...

        def update_E():

            fake = model.sample(noise,noise_logvar)
            real_mu, real_logvar, z_real, rec,flow_log_det_real,xi_real = model(real)
            rec_mu, rec_logvar,z_recon, flow_log_det_recon,xi_recon = model.encode_and_flow(rec.detach())
            fake_mu, fake_logvar,z_fake, flow_log_det_fake,xi_fake = model.encode_and_flow(fake.detach())
            rec_mu, rec_logvar = model.encode(rec.detach())
            fake_mu, fake_logvar = model.encode(fake.detach())
            loss_rec =  model.reconstruction_loss(rec, real, True)

            lossE_real_kl = model.kl_loss(real_mu, real_logvar).mean()-flow_log_det_real.mean()
            lossE_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()-flow_log_det_recon.mean()
            lossE_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()-flow_log_det_fake.mean()
            loss_margin = lossE_real_kl +(torch.relu(opt.m_plus-lossE_rec_kl)+torch.relu(opt.m_plus-lossE_fake_kl)) * 0.5 * opt.weight_neg
            #  + \
            if opt.lambda_me==0:
                T = torch.zeros_like(loss_rec)
            else:
                z_real = model.flow_forward_only(xi_real.detach(), real_logvar.detach())
                T = me_obj(z_real, c)*opt.lambda_me
            # Also, ok might want to add more parametrization of hyper parameters.
            #weight neg should control adversarial objective. Want fakes and (reconstructions?!) to deviate from prior, want reals to be close to prior.
            #Don't know why reconstructions should be adversarial... Might want to rebalance
            lossE = loss_rec  * opt.weight_rec + loss_margin * opt.weight_kl-T
            return lossE,rec,fake,loss_rec,lossE_real_kl,\
                   lossE_rec_kl,lossE_fake_kl,real_logvar,rec_logvar,loss_margin,T,xi_real

        #=========== Update E ================
        if opt.fp_16:
            with autocast():
                lossE,rec,fake,loss_rec,lossE_real_kl,\
                lossE_rec_kl,lossE_fake_kl,\
                real_logvar,rec_logvar,loss_margin,T_loss,xi_real= update_E()
            optimizerG.zero_grad()
            optimizerE.zero_grad()
            scaler.scale(lossE).backward(retain_graph=True)
        else:
            lossE,rec,fake,loss_rec,lossE_real_kl,lossE_rec_kl,\
            lossE_fake_kl,real_logvar\
                ,rec_logvar,loss_margin,T_loss,xi_real = update_E()
            optimizerG.zero_grad()
            optimizerE.zero_grad()
            lossE.backward(retain_graph=True)

        #Backprop everything on everything... NOT! Make sure the FLOW trains only one ONE of the saddlepoint objectives!
        # nn.utils.clip_grad_norm(model.encoder.parameters(), 1.0)

        for m in model.encoder.parameters():
            m.requires_grad=False
        for m in model.flow.parameters(): #Controls whether which objectives apply to flow
            m.requires_grad=False

        #========= Update G ==================
        def update_G():
            rec_mu, rec_logvar, z_recon, flow_log_det_recon,xi_recon = model.encode_and_flow(rec)
            fake_mu, fake_logvar, z_fake, flow_log_det_fake,xi_fake = model.encode_and_flow(fake)
            lossG_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean() - flow_log_det_recon.mean()
            lossG_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean() - flow_log_det_fake.mean()
            lossG = (lossG_rec_kl + lossG_fake_kl) * 0.5 * opt.weight_kl
            return lossG,lossG_rec_kl,lossG_fake_kl

        if opt.fp_16:
            with autocast():
                lossG,lossG_rec_kl,lossG_fake_kl = update_G()
            scaler.scale(lossG).backward()
            # nn.utils.clip_grad_norm(model.decoder.parameters(), 1.0)
            scaler.step(optimizerE)  # .step()
            scaler.step(optimizerG)  # .step()
            scaler.update()
        else:
            lossG,lossG_rec_kl,lossG_fake_kl = update_G()
            lossG.backward()
            # nn.utils.clip_grad_norm(model.decoder.parameters(), 1.0)
            optimizerE.step()
            optimizerG.step()
        for m in model.encoder.parameters():
            m.requires_grad = True
        for m in model.flow.parameters(): #Controls whether which objectives apply to flow
            m.requires_grad=True
        #Write down setups, carefully consider gradient updates to avoid positive feedback loop!

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
        if epoch%opt.save_iter==0:
            save_checkpoint(model, epoch, cur_iter, '',folder_name=f"model{param_suffix}")
        model.train()
        for iteration, (batch,c) in enumerate(train_data_loader, 0):
            if c.sum()<=min_features and (~c).sum()<=min_features:
                continue
            else:
                #--------------train------------
                if epoch < opt.num_vae:
                    train_vae(epoch, iteration, batch,c, cur_iter)
                else:
                    train(epoch, iteration, batch,c, cur_iter)

                cur_iter += 1



if __name__ == "__main__":
    main()    