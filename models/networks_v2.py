import torch
import torch.nn as nn

#Attention eats too much memory

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class CustomSwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)

class _Residual_Block_v2(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block_v2, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.utils.spectral_norm(nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False))
        else:
            self.conv_expand = None

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False))
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = CustomSwish()#nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False))
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = CustomSwish()#nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output

class Encoder_v2(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Encoder_v2, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        self.hdim = hdim
        cc = channels[0]
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(cdim, cc, 5, 1, 2, bias=False)),
            nn.BatchNorm2d(cc),
            CustomSwish(),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for enum,ch in enumerate(channels[1:]):
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block_v2(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2
        # self.main.add_module('Attention_{}'.format(sz),Self_Attn(cc))
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block_v2(cc, cc, scale=1.0))
        self.fc = nn.Linear((cc) * 4 * 4, 2 * hdim)

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar

class Decoder_v2(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Decoder_v2, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        cc = channels[-1]
        self.fc = nn.Sequential(
            nn.Linear(hdim, cc * 4 * 4),
            CustomSwish(),
        )

        sz = 4

        self.main = nn.Sequential()
        for enum,ch in enumerate(channels[::-1]):
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block_v2(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        # self.main.add_module('Attention_{}'.format(sz),Self_Attn(cc))
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block_v2(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        return y


class IntroVAEv2(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(IntroVAEv2, self).__init__()

        self.hdim = hdim

        self.encoder = Encoder_v2(cdim, hdim, channels, image_size)

        self.decoder = Decoder_v2(cdim, hdim, channels, image_size)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return mu, logvar, z, y

    def sample(self, z):
        y = self.decode(z)
        return y

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        y = self.decoder(z)
        return y

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5)  # (batch, 2)
        return v_kl

    def reconstruction_loss(self, prediction, target, size_average=False):
        error = (prediction - target).view(prediction.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=-1)

        if size_average:
            error = error.mean()
        else:
            error = error.sum()

        return error


class Classifier(nn.Module):
    def __init__(self, cdim=3, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Classifier, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        cc = channels[0]
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(cdim, cc, 5, 1, 2, bias=False)),
            nn.BatchNorm2d(cc),
            CustomSwish(),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for enum,ch in enumerate(channels[1:]):
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block_v2(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2
        # self.main.add_module('Attention_{}'.format(sz),Self_Attn(cc))
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block_v2(cc, cc, scale=1.0))
        self.fc = nn.Linear((cc) * 4 * 4, 1)

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        return y.squeeze()
