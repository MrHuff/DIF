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
from main_DIF import parser,record_image,record_scalar,str_to_list,load_model,save_checkpoint
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

parser.add_argument('--n_gpus', type=int, default=1, help='number_of_gpus')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def parallel_main():
    opt = parser.parse_args()
    world_size = opt.n_gpus
    opt.valid_gpu_list = GPUtil.getAvailable(order='memory', limit=opt.n_gpus+1)
    if 5 in opt.valid_gpu_list:
        opt.valid_gpu_list.remove(5)
    listToStr = ','.join([str(elem) for elem in opt.valid_gpu_list])
    print(listToStr)
    os.environ['CUDA_VISIBLE_DEVICES']=listToStr
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    mp.spawn(main,
             args=(world_size,opt),
             nprocs=world_size,
             join=True)


def main(rank,world_size,opt):
    print(torch.__version__)
    # torch.autograd.set_detect_anomaly(True)
    global  model
    print(opt)
    print(f"Running DDP on {rank}.")
    setup(rank,world_size)
    cudnn.benchmark = True
    #--------------build models -------------------------
    single_model = DIF_net(flow_C=opt.C,
                    flow_depth=opt.flow_depth,
                    tanh_flag=opt.tanh_flag,
                    cdim=3,
                    hdim=opt.hdim,
                    channels=str_to_list(opt.channels),
                    image_size=opt.output_height).cuda(rank)

    model = DDP(single_model, device_ids=[rank])

    if opt.pretrained:
        dist.barrier()
        load_model(model, opt.pretrained)
        dist.barrier()

    print(model.module)
            
    optimizerE = optim.Adam(model.module.encoder.parameters(), lr=opt.lr_e)
    optimizerG = optim.Adam(model.module.decoder.parameters(), lr=opt.lr_g)

    if opt.fp_16:
        scaler = GradScaler()

    #-----------------load dataset--------------------------
    train_data = pd.read_csv(opt.class_indicator_file)
    train_list = train_data['file_name'].values.tolist()[:opt.trainsize]
    property_indicator = train_data['class'].values.tolist()[:opt.trainsize]
    #swap out the train files

    assert len(train_list) > 0
    
    train_set = ImageDatasetFromFile_DIF(property_indicator,train_list, opt.dataroot, input_height=None, crop_height=None, output_height=opt.output_height, is_mirror=True)
    sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=(sampler is None),sampler=sampler)
    me_obj = MEstat(J=opt.J,
                    test_nx=len(train_set.property_indicator)-sum(train_set.property_indicator),
                    test_ny=sum(train_set.property_indicator),
                    asymp_n=opt.asymp_n,
                    kernel_type=opt.kernel).cuda(rank)
    min_features = 1./opt.J
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=opt.outf)
    
    start_time = time.time()
            
    cur_iter = 0
    
    def train_vae(epoch, iteration, batch, cur_iter):  
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        real= Variable(batch).cuda(rank) 
                
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time.time()-start_time)
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
        #=========== Update E ================                  
        real_mu, real_logvar, z, rec = model(real) 
        
        loss_rec = model.reconstruction_loss(rec, real, True)
        loss_kl = model.kl_loss(real_mu, real_logvar).mean()
                    
        loss = loss_rec + loss_kl
        
        optimizerG.zero_grad()
        optimizerE.zero_grad()       
        loss.backward()                   
        optimizerE.step() 
        optimizerG.step()
     
        info += 'Rec: {:.4f}, KL: {:.4f}, '.format(loss_rec.data[0], loss_kl.data[0])       
        print(info)
        
        if cur_iter % opt.test_iter is 0:  
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                if cur_iter % 1000 == 0:
                    record_image(writer, [real, rec], cur_iter)   
            else:
                vutils.save_image(torch.cat([real, rec], dim=0).data.cpu(), '{}/image_{}.jpg'.format(opt.outf, cur_iter),nrow=opt.nrow)    
    
    def train(epoch, iteration, batch,c, cur_iter):
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)
            
        batch_size = batch.size(0)

        c = c.cuda(rank)

        noise = Variable(torch.zeros(batch_size, opt.hdim).normal_(0, 1)).cuda(rank) 

        real= Variable(batch).cuda(rank)
        
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time.time()-start_time)
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
        def update_E():

            fake = model.module.sample(noise)
            real_mu, real_logvar, z_real, rec,flow_log_det_real,xi_real = model.module(real)
            rec_mu, rec_logvar,z_recon, flow_log_det_recon,xi_recon = model.module.encode_and_flow(rec.detach())
            fake_mu, fake_logvar,z_fake, flow_log_det_fake,xi_fake = model.module.encode_and_flow(fake.detach())

            loss_rec =  model.module.reconstruction_loss(rec, real, True)

            lossE_real_kl = model.module.kl_loss(real_mu, real_logvar).mean()-flow_log_det_real.mean()
            lossE_rec_kl = model.module.kl_loss(rec_mu, rec_logvar).mean()-flow_log_det_recon.mean()
            lossE_fake_kl = model.module.kl_loss(fake_mu, fake_logvar).mean()-flow_log_det_fake.mean()
            loss_margin = lossE_real_kl + \
                          (F.relu(opt.m_plus-lossE_rec_kl) + \
                          F.relu(opt.m_plus-lossE_fake_kl)) * 0.5 * opt.weight_neg

            lossE = loss_rec  * opt.weight_rec + loss_margin * opt.weight_kl
            return lossE,rec,fake,loss_rec,lossE_real_kl,lossE_rec_kl,lossE_fake_kl,xi_real,xi_recon

        #=========== Update E ================
        if opt.fp_16:
            with autocast():
                lossE,rec,fake,loss_rec,lossE_real_kl,lossE_rec_kl,lossE_fake_kl,xi_real,xi_recon= update_E()
            optimizerG.zero_grad()
            optimizerE.zero_grad()
            scaler.scale(lossE).backward(retain_graph=True)
        else:
            lossE,rec,fake,loss_rec,lossE_real_kl,lossE_rec_kl,lossE_fake_kl,xi_real,xi_recon = update_E()
            optimizerG.zero_grad()
            optimizerE.zero_grad()
            lossE.backward(retain_graph=True)

        def flow_separate_backward():
            z_real = model.module.flow_forward_only(xi_real.detach())
            z_recon = model.module.flow_forward_only(xi_recon.detach())
            T_real = me_obj(z_real,c)
            T_recon = me_obj(z_recon,c)
            return T_real+T_recon
        if opt.fp_16:
            with autocast():
                T_loss = flow_separate_backward()
            scaler.scale(-T_loss * opt.lambda_me).backward()
        else:
            T_loss = flow_separate_backward()
            (-T_loss*opt.lambda_me).backward()

        #Backprop everything on everything...
        # nn.utils.clip_grad_norm(model.encoder.parameters(), 1.0)
        for m in model.module.encoder.parameters():
            m.requires_grad=False
        # for m in model.flow.parameters():
        #     m.requires_grad=False

        #========= Update G ==================
        def update_G():
            rec_mu, rec_logvar, z_recon, flow_log_det_recon,xi_recon = model.module.encode_and_flow(rec)
            fake_mu, fake_logvar, z_fake, flow_log_det_fake,xi_fake = model.module.encode_and_flow(fake)
            lossG_rec_kl = model.module.kl_loss(rec_mu, rec_logvar).mean() - flow_log_det_recon.mean()
            lossG_fake_kl = model.module.kl_loss(fake_mu, fake_logvar).mean()-flow_log_det_fake.mean()
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
        for m in model.module.encoder.parameters():
            m.requires_grad = True
        # for m in model.flow.parameters():
        #     m.requires_grad = True

        info += 'Rec: {:.4f}, '.format(loss_rec.item())
        info += 'Kl_E: {:.4f}, {:.4f}, {:.4f}, '.format(lossE_real_kl.item(),
                                lossE_rec_kl.item(), lossE_fake_kl.item())
        info += 'Kl_G: {:.4f}, {:.4f}, '.format(lossG_rec_kl.item(), lossG_fake_kl.item())
        info += 'ME_flow: {:.4f}'.format(T_loss.item())

        print(info)
        if rank==0:
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
        save_epoch = (epoch//opt.save_iter)*opt.save_iter   
        reload_path = save_checkpoint_parallel(model, save_epoch, 0,rank, '')
        dist.barrier()
        load_model_parallel(model,reload_path,rank)
        dist.barrier()
        model.train()
        for iteration, (batch,c) in enumerate(train_data_loader, 0):
            if c.sum()<=min_features and (~c).sum()<=min_features:
                continue
            else:
                #--------------train------------
                if epoch < opt.num_vae:
                    train_vae(epoch, iteration, batch, cur_iter)
                else:
                    train(epoch, iteration, batch,c, cur_iter)

                cur_iter += 1
    dist.barrier()
    cleanup()

def load_model_parallel(model, load_path,rank):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    model.load_state_dict(
        torch.load(load_path, map_location=map_location))
    return model

def save_checkpoint_parallel(model, epoch, iteration,rank, prefix=""):
    model_out_path = "model/" + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    if rank==0:
        if not os.path.exists("model/"):
            os.makedirs("model/")

        torch.save(model.state_dict(), model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path

if __name__ == "__main__":
    parallel_main()