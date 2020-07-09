

from models.networks import IntroVAE
from IAF.IAF import IAF_flow
import torch
from models.networks_v2 import *
from IAF.layers.utils import accumulate_kl_div, reset_kl_div



class DIF_net(IntroVAE):
    def __init__(self,cdim=3,
                     hdim=512,
                     channels=[64, 128, 256, 512, 512, 512],
                     image_size=256,
                 flow_depth = 3,
                 flow_C=100,
                 tanh_flag=True):
        super(DIF_net, self).__init__(cdim=cdim, hdim=hdim, channels=channels, image_size=image_size)
        self.tanh_flag=tanh_flag
        self.C = flow_C

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        if self.tanh_flag:
            return self.C*torch.tanh(z/self.C)
        else:
            return z

    def sample(self,z):
        if self.tanh_flag:
            return self.decode(self.C * torch.tanh(z / self.C))
        else:
            return self.decode(z)

    def sample_fake_eval(self,n):
        z = torch.randn(n,self.hdim).cuda()
        return self.sample(z)

    def get_latent(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

class DIF_net_flow(IntroVAE):
    def __init__(self,cdim=3,
                     hdim=512,
                     channels=[64, 128, 256, 512, 512, 512],
                     image_size=256,
                 flow_depth = 3,
                 flow_C=100,
                 tanh_flag=True):
        super(DIF_net_flow, self).__init__(cdim=cdim, hdim=hdim, channels=channels, image_size=image_size)
        self.tanh_flag=tanh_flag
        self.C = flow_C
        self.flow = IAF_flow(hdim,flow_depth,tanh_flag,flow_C)

    def forward(self, x):
        mu, logvar = self.encode(x)
        xi,z,flow_log_det = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return mu, logvar, z, y, flow_log_det,xi

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(mu)
        xi = eps.mul(std).add_(mu)
        z,log_det = self.flow(xi,logvar)
        return xi,z,log_det

    def flow_forward_only(self,xi,logvar=None):
        output,_ = self.flow(xi, logvar)
        return output

    def encode_and_flow(self,x):
        mu, logvar = self.encode(x)
        xi,z,flow_log_det = self.reparameterize(mu, logvar)
        return mu, logvar, z, flow_log_det,xi

    def get_latent(self,x):
        return self.encode_and_flow(x)

    def sample(self,xi,logvar):
        with torch.no_grad():
            z,_ = self.flow(xi,logvar)
        return self.decode(z.detach())

    def sample_fake_eval(self, n):
        z = torch.randn(n, self.hdim).cuda()
        logvar = torch.zeros_like(z)
        return self.sample(z,logvar)



class DIF_netv2(IntroVAEv2):
    def __init__(self,cdim=3,
                     hdim=512,
                     channels=[64, 128, 256, 512, 512, 512],
                     image_size=256,
                 flow_depth = 3,
                 flow_C=100,
                 tanh_flag=True):
        super(DIF_netv2, self).__init__(cdim=cdim, hdim=hdim, channels=channels, image_size=image_size)
        self.tanh_flag=tanh_flag
        self.C = flow_C

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        if self.tanh_flag:
            return self.C*torch.tanh(z/self.C)
        else:
            return z

    def sample(self,z):
        if self.tanh_flag:
            return self.decode(self.C * torch.tanh(z / self.C))
        else:
            return self.decode(z)

    def sample_fake_eval(self,n):
        z = torch.randn(n,self.hdim).cuda()
        return self.sample(z)

    def get_latent(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

class DIF_net_flow_v2(IntroVAEv2):
    def __init__(self,cdim=3,
                     hdim=512,
                     channels=[64, 128, 256, 512, 512, 512],
                     image_size=256,
                 flow_depth = 3,
                 flow_C=100,
                 tanh_flag=True):
        super(DIF_net_flow, self).__init__(cdim=cdim, hdim=hdim, channels=channels, image_size=image_size)
        self.tanh_flag=tanh_flag
        self.C = flow_C
        self.flow = IAF_flow(hdim,flow_depth,tanh_flag,flow_C)

    def forward(self, x):
        mu, logvar = self.encode(x)
        xi,z,flow_log_det = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return mu, logvar, z, y, flow_log_det,xi

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(mu)
        xi = eps.mul(std).add_(mu)
        z,log_det = self.flow(xi,logvar)
        return xi,z,log_det

    def flow_forward_only(self,xi,logvar=None):
        output,_ = self.flow(xi, logvar)
        return output

    def encode_and_flow(self,x):
        mu, logvar = self.encode(x)
        xi,z,flow_log_det = self.reparameterize(mu, logvar)
        return mu, logvar, z, flow_log_det,xi

    def get_latent(self,x):
        return self.encode_and_flow(x)

    def sample(self,xi,logvar):
        with torch.no_grad():
            z,_ = self.flow(xi,logvar)
        return self.decode(z.detach())

    def sample_fake_eval(self, n):
        z = torch.randn(n, self.hdim).cuda()
        logvar = torch.zeros_like(z)
        return self.sample(z,logvar)