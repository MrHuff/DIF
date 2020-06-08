import torch as t
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_normal,xavier_normal_
from torch.nn.parameter import Parameter

class IAF_no_h(nn.Module):
    def __init__(self, latent_size, depth,tanh_flag_h=False,C=100):
        super(IAF_no_h, self).__init__()
        self.C = C
        self.depth = depth
        self.z_size = latent_size
        self.tanh_op = nn.Tanh()
        self.flag = tanh_flag_h
        self.s_list = nn.ModuleList(
            [nn.Sequential(AutoregressiveLinear(self.z_size , self.z_size), nn.ELU()) for i
             in range(depth)])
        self.m_list = nn.ModuleList(
            [nn.Sequential(AutoregressiveLinear(self.z_size , self.z_size), nn.ELU()) for i
             in range(depth)])

    def forward(self, z):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :return: An float tensor with shape of [batch_size, z_size] and log det value of the IAF mapping Jacobian
        """
        log_det = 0
        for i in range(self.depth):
            m = self.m_list[i](z)
            s = self.s_list[i](z)
            z = s * z + (1 - s) * m
            log_det = log_det - s.log().sum(1)
        if self.flag:
            z = self.tanh_op(z/self.C)*self.C
        return z, -log_det

    def flow_pass_only(self,z):
        for i in range(self.depth):
            m = self.m_list[i](z)
            s = self.s_list[i](z)
            z = s * z + (1 - s) * m
        if self.flag:
            z = self.tanh_op(z / self.C) * self.C
        return z


class IAF(nn.Module):
    def __init__(self, latent_size, h_size,depth,tanh_flag=False,C=100):
        super(IAF, self).__init__()
        self.depth = depth
        self.z_size = latent_size
        self.h_size = h_size
        self.tanh_op = nn.Tanh()
        self.flag = tanh_flag
        self.h = Highway(self.h_size, 3, nn.ELU())
        self.C = C
        self.z_size = latent_size
        self.s_list = nn.ModuleList([nn.Sequential(AutoregressiveLinear(self.z_size+self.h_size, self.z_size),nn.ELU()) for i in range(depth)])
        self.m_list = nn.ModuleList([nn.Sequential(AutoregressiveLinear(self.z_size+self.h_size, self.z_size),nn.ELU()) for i in range(depth)])

    def forward(self, z, h):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :param h: An float tensor with shape of [batch_size, h_size]
        :return: An float tensor with shape of [batch_size, z_size] and log det value of the IAF mapping Jacobian
        """
        h = self.h(h)
        log_det = 0
        for i in range(self.depth):
            input = t.cat([z, h], 1)
            m = self.m_list[i](input)
            s = self.s_list[i](input)
            z = s*z+(1-s)*m
            log_det = log_det - s.log().sum(1)
        if self.flag:
            z = self.tanh_op(z/self.C)*self.C
        return z, -log_det



class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.utils.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.utils.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.utils.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ f(G(x)) + (1 - σ(x)) ⨀ Q(x) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x



class AutoregressiveLinear(nn.Module):
    def __init__(self, in_size, out_size, bias=True, ):
        super(AutoregressiveLinear, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.weight = Parameter(t.Tensor(self.in_size, self.out_size))

        if bias:
            self.bias = Parameter(t.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = xavier_normal_(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return t.addmm(self.bias, input, self.weight.tril(-1))

        output = input @ self.weight.tril(-1)
        if self.bias is not None:
            output += self.bias
        return output