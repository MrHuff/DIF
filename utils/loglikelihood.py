import torch
from utils.model_utils import *
#Class conditional loglikelihood!

def elbo_recon(prediction,target):
    error = (prediction - target).view(prediction.size(0), -1)
    error = error ** 2
    error = torch.sum(error, dim=-1)
    return error

def calculate_ELBO(model,real_images):
    with torch.no_grad():
        real_mu, real_logvar, z_real, rec = model(real_images)
        loss_rec = elbo_recon(rec,real_images)
        loss_kl = model.kl_loss(real_mu, real_logvar)
        ELBO = loss_rec+loss_kl
    return -ELBO.squeeze()

def estimate_loglikelihoods(dataloader_test, model,s=1000):
    _loglikelihood_estimates = []
    _elbo_estimates = []
    _class = []
    tensor_s = torch.tensor(s).float()
    with torch.no_grad():
        for iteration, (batch, c) in enumerate(tqdm.tqdm(dataloader_test)):
            _elbo = []
            for i in tqdm.trange(s):
                with autocast():
                    ELBO = calculate_ELBO(model,batch.cuda())
                _elbo.append(ELBO)
            _elbo_estimates.append(ELBO)
            likelihood_est = torch.stack(_elbo,dim=1)
            # print(torch.logsumexp(likelihood_est,dim=1).cpu()-torch.log(tensor_s))
            _loglikelihood_estimates.append(torch.logsumexp(likelihood_est,dim=1).cpu()-torch.log(tensor_s))
            _class.append(c)
        _elbo_estimates = torch.cat(_elbo_estimates,dim=0)
        _class = torch.cat(_class,dim=0)
        _loglikelihood_estimates = torch.cat(_loglikelihood_estimates,dim=0)
        return _loglikelihood_estimates,_elbo_estimates,_class

def calculate_metrics(_loglikelihood_estimates,_elbo_estimates,_class):
    with torch.no_grad():
        loglikelihood_estimate = _loglikelihood_estimates.mean(0)
        ELBO = _elbo_estimates.mean()
        loglikelihood_estimate_A =_loglikelihood_estimates[~_class].mean(0)
        loglikelihood_estimate_B = _loglikelihood_estimates[_class].mean(0)
        ELBO_A = _elbo_estimates[~_class].mean()
        ELBO_B = _elbo_estimates[_class].mean()

    return loglikelihood_estimate.item(),ELBO.item(),loglikelihood_estimate_A.item(),loglikelihood_estimate_B.item(),ELBO_A.item(),ELBO_B.item()


