from IAF.flows.iaf import IAF_mod
from torch import nn
import torch
class IAF_flow(nn.Module):
    def __init__(self, dim, n_flows,tanh_flag,C=100):
        super().__init__()
        self.flow = nn.ModuleList([
            IAF_mod(dim,dim,dim) for _ in range(n_flows)
        ])
        self.C = C
        self.tanh_flag = tanh_flag

    def forward(self, z0,h=None):
        log_det = 0
        zk = z0
        for f in self.flow:
            zk,ld = f(zk,h)
            log_det= log_det+ld
        if self.tanh_flag:
            return self.C*torch.tanh(zk/self.C),log_det
        else:
            return zk,log_det

