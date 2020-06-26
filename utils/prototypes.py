from utils.model_utils import *
from gpytorch.kernels import RBFKernel,Kernel
import torch
import scipy.stats as stats

class witness_generation(torch.nn.Module):
    def __init__(self,hdim,n_witnesses,latents,c,coeff=1e-2,init_type='randn'):
        super(witness_generation, self).__init__()
        self.hdim = hdim
        self.n_witnesses= n_witnesses
        if init_type=='randn':
            init_vals = torch.randn(n_witnesses,hdim)
        self.T = torch.nn.Parameter(init_vals,requires_grad=True)
        self.register_buffer('X_data',latents[~c,:])
        self.register_buffer('Y_data',latents[c,:])
        self.ls = self.get_median_ls(latents)
        self.kernel = RBFKernel()
        self.kernel.raw_lengthscale = torch.nn.Parameter(self.ls, requires_grad=True)
        self.diag = torch.nn.Parameter(coeff*torch.eye(n_witnesses),requires_grad=False).float() #Tweak this badboy for FP_16

    def get_median_ls(self, X): #Super LS and init value sensitive wtf
        with torch.no_grad():
            self.kernel_base = Kernel()
            if X.shape[0]>5000:
                idx = torch.randperm(5000)
                X = X[idx,:]
            d = self.kernel_base.covar_dist(X, X)
            return torch.sqrt(torch.median(d[d > 0])).unsqueeze(0)

    def optimize_kernel(self):
        self.T.requires_grad=False
        self.kernel.raw_lengthscale.requires_grad=True

    def optimize_witness(self):
        self.T.requires_grad = True
        self.kernel.raw_lengthscale.requires_grad = False

    def calculate_hotelling(self,X):
        k_X = self.kernel(X,self.T).evaluate()
        x_bar = torch.mean(k_X,0)
        k_X = k_X - x_bar
        cov_X = torch.mm(k_X.t(),k_X)
        return cov_X,x_bar

    def forward(self):
        cov_X,x_bar = self.calculate_hotelling(self.X)
        cov_Y,y_bar = self.calculate_hotelling(self.Y)
        pooled = 1/(self.nx+self.ny-2) * (cov_X+ cov_Y)
        z = (x_bar-y_bar).unsqueeze(1)
        inv_z,_ = torch.solve(z,pooled + self.diag)
        test_statistic = -self.nx*self.ny/(self.nx + self.ny) *torch.sum(z*inv_z)
        return test_statistic

    def get_pval_test(self,stat):
        pvalue = stats.chi2.sf(stat, self.n_witnesses)
        return pvalue

    def return_witnesses(self):
        return self.T.detach()

def training_loop_witnesses(hdim,
                            n_witnesses,
                            latents,
                            c,
                            coeff=1e-2,
                            init_type='randn',
                            cycles=40,
                            its = 50,
                            device="cpu"):

    witness_obj = witness_generation(hdim,n_witnesses,latents,c,coeff=coeff,init_type=init_type).to(device)
    optimizer = torch.optim.Adam(witness_obj.parameters(), lr=1e-3)
    lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    for i in range(cycles):
        for t in [True,False]:
            if t:
                witness_obj.optimize_kernel()
            else:
                witness_obj.optimize_witness()
            for j in range(its):
                tst_statistic = witness_obj()
                optimizer.zero_grad()
                tst_statistic.backward()
                optimizer.step()
                lrs.step(tst_statistic)
            print(f'test statistic: {tst_statistic.item()}')
    pval = witness_obj.get_pval_test(tst_statistic.item())
    return witness_obj,pval
# with torch.no_grad():
#     list_xi = []
#     list_z = []
#     list_c = []
#     for iteration,(batch,c) in enumerate(train_data_loader,0):
#
#         if iteration<20:
#             c = c.cuda(base_gpu)
#             batch = batch.cuda(base_gpu)
#             list_c.append(c)
#             real_mu, real_logvar, z_real, rec = model(batch)
#             # real_mu, real_logvar, z_real, rec, flow_log_det_real, xi_real = model(batch)
#             list_z.append(z_real)
#             list_xi.append(real_mu)
#         else:
#             break
#     big_c = torch.cat(list_c,dim=0)
#     big_xi = torch.cat(list_xi,dim=0)
#     big_z = torch.cat(list_z,dim=0)
#     x_class_xi, y_class_xi = subset_latents(big_xi,big_c)
#     make_binary_class_umap_plot(x_class_xi,y_class_xi,opt.outf,cur_iter,'xi_plot')
#     x_class_z, y_class_z = subset_latents(big_z,big_c)
#     make_binary_class_umap_plot(x_class_z,y_class_z,opt.outf,cur_iter,'z_plot')

#calculate FID for prototypes

