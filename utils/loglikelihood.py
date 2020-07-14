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
    for iteration, (batch, c) in enumerate(tqdm.tqdm(dataloader_test)):
        _elbo = []
        for i in tqdm.trange(s):
            with autocast():
                ELBO = calculate_ELBO(model,batch.cuda())
            _elbo.append(ELBO)
        _elbo_estimates.append(ELBO)
        likelihood_est = torch.stack(_elbo,dim=1)
        _loglikelihood_estimates.append(likelihood_est.mean(-1))
        _class.append(c)
    _elbo_estimates = torch.cat(_elbo_estimates,dim=0)
    _class = torch.cat(_class,dim=0)
    _loglikelihood_estimates = torch.cat(_loglikelihood_estimates,dim=0)
    return _loglikelihood_estimates,_elbo_estimates,_class

def loglikelihood_est(_loglikelihood_estimates):
    return torch.logsumexp(_loglikelihood_estimates, 0).cpu() - torch.log(torch.tensor(_loglikelihood_estimates.shape[0]).float())

def calculate_metrics(_loglikelihood_estimates,_elbo_estimates,_class):
    with torch.no_grad():
        loglikelihood_estimate = loglikelihood_est(_loglikelihood_estimates)
        ELBO = _elbo_estimates.mean()
        loglikelihood_estimate_A = loglikelihood_est(_loglikelihood_estimates[~_class])
        loglikelihood_estimate_B = loglikelihood_est(_loglikelihood_estimates[_class])
        ELBO_A = _elbo_estimates[~_class].mean()
        ELBO_B = _elbo_estimates[_class].mean()

    return loglikelihood_estimate.item(),ELBO.item(),loglikelihood_estimate_A.item(),loglikelihood_estimate_B.item(),ELBO_A.item(),ELBO_B.item()


