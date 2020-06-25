import torch
from torch import nn
from gpytorch.kernels import LinearKernel,MaternKernel,RBFKernel,Kernel
from torch.nn.modules.loss import _Loss

class Log1PlusExp(torch.autograd.Function):
    """Implementation of x ↦ log(1 + exp(x))."""
    @staticmethod
    def forward(ctx, x):
        exp = x.exp()
        ctx.save_for_backward(x)
        y = exp.log1p()
        return x.where(torch.isinf(exp),y.half() if x.type()=='torch.cuda.HalfTensor' else y )

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (-x).exp().half() if x.type()=='torch.cuda.HalfTensor' else (-x).exp()
        return grad_output / (1 + y)

class stableBCEwithlogits(_Loss):
    def __init__(self, reduction='mean'):
        super(stableBCEwithlogits, self).__init__(reduction=reduction)
        self.f = Log1PlusExp.apply

    def forward(self, x, y):
        return torch.mean(self.f(x)-x*y)

class linear_benchmark(nn.Module):
    def __init__(self,d):
        super(linear_benchmark, self).__init__()
        self.register_buffer('w',torch.ones(d))
        self.objective = stableBCEwithlogits()

    def forward(self,data,c,debug_xi = None):
        X = data[~c, :]
        Y = data[c, :]
        target = torch.cat([torch.zeros(X.shape[0]),torch.ones(Y.shape[0])]).to(X.device)
        data = torch.cat([X,Y])
        pred = (data@self.w).squeeze()
        return -self.objective(pred,target)

class MEstat(nn.Module):

    def __init__(self,J,ls=10,test_nx=1,test_ny=1,asymp_n=-1,kernel_type = 'rbf',linear_var=1e-3):
        super(MEstat, self).__init__()
        print(ls)
        self.ratio = J
        self.hotelling = False
        self.kernel_type = kernel_type
        if kernel_type=='hotelling': #Regularization fixes it...

            self.hotelling = True
            self.coeff = min(min(test_nx, test_ny) ** asymp_n, 1e-2)
        else:
            if kernel_type=='rbf':
                self.kernel_X = RBFKernel()
                self.kernel_X.raw_lengthscale = nn.Parameter(torch.tensor([ls]).float(), requires_grad=False)
            elif kernel_type=='linear':
                self.kernel_X = LinearKernel()
                self.kernel_X._set_variance(linear_var)
            elif kernel_type=='matern':
                self.kernel_X = MaternKernel(nu=2.5)
                self.kernel_X.raw_lengthscale = nn.Parameter(torch.tensor([ls]).float(), requires_grad=False)

            self.coeff = min(min(test_nx, test_ny) ** asymp_n, 1e-5)
        self.kernel_base = Kernel()

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(X,X)
            return torch.sqrt(torch.median(d[d > 0]))
    @staticmethod
    def cov(m, rowvar=False):
        '''Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            m: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        '''
        if m.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if m.dim() < 2:
            m = m.view(1, -1)
        if not rowvar and m.size(0) != 1:
            m = m.t()
        # m = m.type(torch.double)  # uncomment this line if desired
        m_mean = torch.mean(m, dim=1, keepdim=True)
        m = m - m_mean
        return  m.matmul(m.t()).squeeze(),m_mean.squeeze()

    def calculate_hotelling(self, X):
        cov_X,x_bar = self.cov(X)
        return cov_X,x_bar,0,0

    def get_sample_witness(self,X,Y):
        n_x = X.shape[0]
        n_y = Y.shape[0]
        idx = torch.randperm(n_x)
        idy = torch.randperm(n_y)
        J_x = round(n_x*self.ratio)
        J_y = round(n_y*self.ratio)
        T_x, T_y = X[idx[:J_x], :].detach(), Y[idy[:J_y], :].detach()
        X,Y = X[idx[J_x:], :], Y[idy[J_y:], :]
        return T_x,T_y,X,Y

    def get_umap_stuff(self,X,Y,T):
        kX = self.kernel_X(X, T).evaluate()
        kY = self.kernel_X(Y,T).evaluate()
        return kX,kY,torch.cat([kX,kY],dim=0)

    def forward_plain(self,X,Y,T,n_x,n_y):
        if not self.hotelling:
            cov_X,x_bar,k_X,kX = self.calculate_ME_hotelling(X, T)
            cov_Y,y_bar,k_Y,kY = self.calculate_ME_hotelling(Y, T)
        else:
            cov_X, x_bar, k_X, kX = self.calculate_hotelling(X)
            cov_Y, y_bar, k_Y, kY = self.calculate_hotelling(Y)
        pooled = 1. / (n_x + n_y - 2.) * (cov_X + cov_Y)
        z = torch.unsqueeze(x_bar - y_bar, 1)
        inv_z,_ = torch.solve(z,pooled.float() + self.coeff*torch.eye(pooled.shape[0]).float().to(pooled.device))
        test_statistic = n_x * n_y / (n_x + n_y) * torch.sum(z * inv_z)
        return test_statistic

    def forward(self,data,c,debug_xi_hat=None):
        X = data[~c,:]
        Y = data[c,:]
        tmp_dev = X.device
        if not self.hotelling:
            T_x,T_y,X,Y = self.get_sample_witness(X,Y)
            n_x  = X.shape[0]
            n_y  = Y.shape[0]
            T = torch.cat([T_x, T_y],dim=0)
            if not self.kernel_type=='linear':
                _tmp = torch.cat([X, Y], dim=0).detach()
                with torch.no_grad():
                    sig = self.get_median_ls(_tmp)
                    self.kernel_X.raw_lengthscale = nn.Parameter(sig.unsqueeze(-1).to(tmp_dev),requires_grad=False)  # Use old setup?!??!?!?!
            else:
                _tmp  = torch.tensor(0)
                sig=0
            cov_X,x_bar,k_X,kX = self.calculate_ME_hotelling(X, T)
            cov_Y,y_bar,k_Y,kY = self.calculate_ME_hotelling(Y, T)
        else:
            _tmp = 0
            n_x  = X.shape[0]
            n_y  = Y.shape[0]
            cov_X,x_bar,k_X,kX = self.calculate_hotelling(X)
            cov_Y,y_bar,k_Y,kY = self.calculate_hotelling(Y)
        pooled = 1./(n_x+n_y-2.) * cov_X +  cov_Y*1./(n_x+n_y-2.)
        z = torch.unsqueeze(x_bar-y_bar,1)
        inv_z,_ = torch.solve(z.float(),pooled.float() + self.coeff*torch.eye(pooled.shape[0]).float().to(tmp_dev))
        test_statistic = n_x*n_y/(n_x + n_y) * torch.sum(z*inv_z)

        if test_statistic.data ==0 or test_statistic==float('inf') or test_statistic!=test_statistic: #The lengthscale be fucking me...
            print(test_statistic)
            print(x_bar)
            print(y_bar)
            print(inv_z)
            print(cov_X)
            print(cov_Y)
            print(k_X)
            print(k_Y)
            print(kX)
            print(kY)
            print(_tmp.min(),_tmp.max())
            print(sig)
            print(n_x*n_y/(n_x + n_y))
            print(pooled)
        return test_statistic

    def calculate_ME_hotelling(self, X, T):
        kX = self.kernel_X(X, T).evaluate()
        x_bar = torch.mean(kX, dim=0)
        k_X = kX - x_bar
        cov_X = k_X.t() @ k_X
        return cov_X, x_bar, k_X, kX

