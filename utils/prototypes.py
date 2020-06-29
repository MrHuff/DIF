from utils.model_utils import *
from gpytorch.kernels import RBFKernel,Kernel
import torch
import scipy.stats as stats
import numpy as np
class witness_generation(torch.nn.Module):
    def __init__(self,hdim,n_witnesses,X,Y,coeff=1e-2,init_type='randn'):
        super(witness_generation, self).__init__()
        self.hdim = hdim
        self.n_witnesses= n_witnesses
        if init_type=='randn':
            init_vals = torch.randn(n_witnesses,hdim)
        self.T = torch.nn.Parameter(init_vals,requires_grad=True)
        self.register_buffer('X',X)
        self.register_buffer('Y',Y)
        self.nx = self.X.shape[0]
        self.ny = self.Y.shape[0]
        self.ls = self.get_median_ls(torch.cat([self.X,self.Y]))
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

    def forward(self,X=None,Y=None):
        if X is None:
            X = self.X
            nx = self.nx
        else:
            nx = X.shape[0]
        if Y is None:
            Y = self.Y
            ny = self.ny
        else:
            ny = Y.shape[0]
        cov_X,x_bar = self.calculate_hotelling(X)
        cov_Y,y_bar = self.calculate_hotelling(Y)
        pooled = 1/(nx+ny-2) * (cov_X+ cov_Y)
        z = (x_bar-y_bar).unsqueeze(1)
        inv_z,_ = torch.solve(z,pooled + self.diag)
        test_statistic = -nx*ny/(nx + ny) *torch.sum(z*inv_z)
        return test_statistic

    def get_pval_test(self,stat):
        pvalue = stats.chi2.sf(stat, self.n_witnesses)
        return pvalue

    def return_witnesses(self):
        return self.T.detach()

def training_loop_witnesses(hdim,
                            n_witnesses,
                            train_latents,
                            c_train,
                            test_latents,
                            c_test,
                            coeff=1e-2,
                            init_type='randn',
                            cycles=40,
                            its = 50):
    X = train_latents[~c_train,:]
    Y = train_latents[c_train,:]
    tr_nx = round(X.shape[0]*0.9)
    tr_ny = round(Y.shape[0]*0.9)

    val_X = X[tr_nx:,:]
    val_Y = Y[tr_ny:,:]
    test_X = test_latents[~c_test,:]
    test_Y = test_latents[c_test,:]
    witness_obj = witness_generation(hdim, n_witnesses, X[:tr_nx,:], Y[:tr_ny,:], coeff=coeff, init_type=init_type).cuda()
    optimizer = torch.optim.Adam(witness_obj.parameters(), lr=1e-1)
    lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
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
            with torch.no_grad():
                val_stat_test = witness_obj(val_X, val_Y)
            print(f'test statistic: {val_stat_test.item()}')
            lrs.step(val_stat_test.item())
    with torch.no_grad():
        tst_stat_test = witness_obj(test_X,test_Y)
    pval = witness_obj.get_pval_test(tst_stat_test.item())
    return witness_obj,pval


