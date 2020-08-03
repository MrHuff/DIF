from utils.model_utils import *
from gpytorch.kernels import RBFKernel,Kernel
import torch
import scipy.stats as stats
import numpy as np


def init_locs_2randn(X_torch,Y_torch, n_test_locs, subsample=10000):
    """Fit a Gaussian to each dataset and draw half of n_test_locs from
    each. This way of initialization can be expensive if the input
    dimension is large.

    """

    X = X_torch.cpu().numpy()
    Y = Y_torch.cpu().numpy()
    n = min(X.shape[0],Y.shape[0])
    seed = np.random.randint(0,1000)
    np.random.seed(seed)
        # Subsample X, Y if needed. Useful if the data are too large.
    if n > subsample:
        X = X[np.random.choice(X.shape[0], subsample, replace=False),:]
        Y = Y[np.random.choice(Y.shape[0], subsample, replace=False),:]

    d = X.shape[1]
    if d > 5000:
        Tx = np.mean(X, axis=0)
        Ty = np.mean(Y, axis=0)
        T0 = np.vstack((Tx, Ty))
    else:
        # fit a Gaussian to each of X, Y
        mean_x = np.mean(X, 0)
        mean_y = np.mean(Y, 0)
        cov_x = np.cov(X.T)
        [Dx, Vx] = np.linalg.eig(cov_x + 1e-3 * np.eye(d))
        Dx = np.real(Dx)
        Vx = np.real(Vx)
        # a hack in case the data are high-dimensional and the covariance matrix
        # is low rank.
        Dx[Dx <= 0] = 1e-3

        # shrink the covariance so that the drawn samples will not be so
        # far away from the data
        eig_pow = 1.0  # 1.0 = not shrink
        reduced_cov_x = Vx.dot(np.diag(Dx ** eig_pow)).dot(Vx.T) + 1e-3 * np.eye(d)
        cov_y = np.cov(Y.T)
        [Dy, Vy] = np.linalg.eig(cov_y + 1e-3 * np.eye(d))
        Vy = np.real(Vy)
        Dy = np.real(Dy)
        Dy[Dy <= 0] = 1e-3
        reduced_cov_y = Vy.dot(np.diag(Dy ** eig_pow).dot(Vy.T)) + 1e-3 * np.eye(d)
        # integer division
        Jx = n_test_locs//2
        Jy = n_test_locs - Jx

        # from IPython.core.debugger import Tracer
        # t = Tracer()
        # t()
        assert Jx + Jy == n_test_locs, 'total test locations is not n_test_locs'
        Tx = np.random.multivariate_normal(mean_x, reduced_cov_x, Jx)
        Ty = np.random.multivariate_normal(mean_y, reduced_cov_y, Jy)
        T0 = np.vstack((Tx, Ty))

    return T0

class witness_generation(torch.nn.Module):
    def __init__(self,hdim,n_witnesses,X,Y,coeff=1e-5,init_type='randn'):
        super(witness_generation, self).__init__()
        self.hdim = hdim
        self.n_witnesses= n_witnesses

        if init_type=='randn':
            init_vals = torch.randn(n_witnesses,hdim)
        elif init_type=='median_noise':
            x = X.mean(dim=0)
            y = Y.mean(dim=0)
            cat = torch.cat([x.unsqueeze(0).repeat(n_witnesses//2,1),y.unsqueeze(0).repeat(n_witnesses//2,1)])
            init_vals = cat + torch.randn_like(cat)

        elif init_type=='gaussian_fit':
            init_vals = torch.tensor(init_locs_2randn(X,Y,n_test_locs=n_witnesses,subsample=200000)).float()


        self.T = torch.nn.Parameter(init_vals,requires_grad=True)
        self.register_buffer('X',X)
        self.register_buffer('Y',Y)
        self.nx = self.X.shape[0]
        self.ny = self.Y.shape[0]
        self.ls = self.get_median_ls(torch.Tensor(init_vals))
        print(self.ls)
        self.kernel = RBFKernel()
        self.kernel.raw_lengthscale = torch.nn.Parameter(self.ls, requires_grad=True)
        self.diag = torch.nn.Parameter(coeff*torch.eye(n_witnesses),requires_grad=False).float() #Tweak this badboy for FP_16

    def get_median_ls(self, X): #Super LS and init value sensitive wtf
        with torch.no_grad():
            self.kernel_base = Kernel()
            if X.shape[0]>10000:
                idx = torch.randperm(10000)
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

def training_loop_witnesses( #control ls-updates, as we change the RKHS the optimization surface changes. Might get stuck in local optimz
        save_path,
        hdim,
                            n_witnesses,
                            train_latents,
                            c_train,
                            test_latents,
                            c_test,
                            coeff=1e-5,
                            init_type='randn',
                            cycles=50,
                            its = 36,
                            patience=5):
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
    counter = 0
    best = np.inf
    print(witness_obj.T)
    for i in range(cycles):
        for t in [True,False]:
            if t:
                witness_obj.optimize_kernel()
                train_its = its//2
            else:
                witness_obj.optimize_witness()
                train_its = its
            for j in range(train_its):
                tst_statistic = witness_obj()
                optimizer.zero_grad()
                tst_statistic.backward()
                optimizer.step()
                print(f'train err: {-tst_statistic.item()}')
            with torch.no_grad():
                val_stat_test = witness_obj(val_X, val_Y)
                tst_stat_test = witness_obj(test_X, test_Y)

            print(f'test statistic val data: {val_stat_test.item()}')
            print(f'test statistic test data: {tst_stat_test.item()}')
            lrs.step(val_stat_test.item())
            if val_stat_test.item()<best:
                best = val_stat_test.item()
                torch.save(witness_obj.state_dict(), save_path + 'witness_object.pth')
                print(f'best score: {val_stat_test.item()}')
                print(f'best score: {tst_stat_test.item()}')
                print(witness_obj.T)

            else:
                counter+=1
        if counter>=patience:
            print('no more improvement breaking')
            break
    with torch.no_grad():
        witness_obj.load_state_dict(torch.load(save_path+'witness_object.pth'))
        tst_stat_test = witness_obj(test_X,test_Y)
    print(witness_obj.T)
    pval = witness_obj.get_pval_test(-tst_stat_test.item())
    return witness_obj,pval


