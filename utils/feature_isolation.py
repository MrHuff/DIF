import torch
from sklearn import metrics
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt

def weights_histogram(linear_model,save_path,alpha):
    w = linear_model.linear.weight.detach().cpu().numpy()
    plt.bar([i for i in range(w.shape[1])], w.squeeze())
    plt.savefig(save_path+f'w_plot_{alpha}.png')
    plt.clf()

def auc_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.squeeze().float() > 0.5).cpu().float().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(Y.cpu().numpy(), y_pred, pos_label=1)
        auc =  metrics.auc(fpr, tpr)
        return auc

class regression_dataset(Dataset):
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:],self.Y[idx]

class lasso_regression(torch.nn.Module):
    def __init__(self,in_dim,o_dim):
        super(lasso_regression,self).__init__()
        self.linear = torch.nn.Linear(in_features=in_dim,out_features= o_dim,bias=True)
    def lasso_term(self):
        return torch.norm(self.linear.weight,p=1)

    def forward(self,x):
        return self.linear(x)

def lasso_train(save_path,data_train,c_train,data_test,c_test,reg_parameter,lr,epochs,bs_rate=1.0,patience = 10):
    train_idx = np.random.randn(data_train.shape[0])<=0.9
    X_train = data_train[train_idx,:]
    Y_train = c_train[train_idx]
    X_val = data_train[~train_idx,:]
    Y_val = c_train[~train_idx]
    X_test = data_test
    Y_test = c_test
    bs = round(bs_rate*X_train.shape[0])
    pos_weight = (Y_train.shape[0]-Y_train.sum().float())/Y_train.sum().float()
    pos_weight = pos_weight.cuda()
    model = lasso_regression(in_dim=X_train.shape[1], o_dim=1).cuda()
    opt = torch.optim.Adam(params=model.parameters(),lr=lr)
    dataset = regression_dataset(X_train,Y_train)
    objective = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader  = DataLoader(dataset,batch_size = bs)
    lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=2)
    best = -np.inf
    counter = 0
    for i in range(epochs):
        for j,(X_batch,y_batch) in enumerate(loader):
            y_batch = y_batch.cuda()
            opt.zero_grad()
            y_pred = model(X_batch)
            e = objective(y_pred.squeeze(),y_batch.float().squeeze()) + reg_parameter*model.lasso_term() #lasso term screwing things up
            e.backward()
            opt.step()
        with torch.no_grad():
            val_pred = model(X_val)
            val_auc = auc_check(val_pred, Y_val)
            test_pred = model(X_test)
            test_auc = auc_check(test_pred, Y_test)
            print(f'val auc: {val_auc}')
            print(f'test auc: {test_auc}')
            lrs.step(-val_auc)
            if val_auc>best:
                best = val_auc
                counter = 0
                print(f'best auc: {val_auc}')
                print(f'best auc: {test_auc}')
                torch.save(model.state_dict(), save_path + f'lasso_latents_{reg_parameter}.pth')
            else:
                counter+=1
            if counter>=patience:
                print('no more improvement breaking')
                break
    return model,test_auc

def get_feature_sparsity(model):
    w = model.linear.weight.detach().cpu().numpy()
    w = np.abs(w)
    largest_feature = np.max(w)
    sparse_threshold = 0.1*largest_feature
    sparsity_level = (w>=sparse_threshold).sum()/w.shape[1]
    return sparsity_level
