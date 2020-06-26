import torch
from sklearn import metrics
from torch.utils.data import Dataset,DataLoader

def auc_check(model,X,Y):
    with torch.no_grad():
        y_pred,_ = model(X)
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

def lasso_train(data_train,c_train,data_test,c_test,reg_parameter,lr,epochs,device):
    X_train = data_train
    Y_train = c_train
    X_test = data_test
    Y_test = c_test

    pos_weight = (Y_train.numel()-Y_train.sum())/Y_train.sum()
    model = lasso_regression(in_dim=X_train.shape[1], o_dim=1).to(device)
    opt = torch.optim.Adam(params=model.parameters(),lr=lr)
    dataset = regression_dataset(X_train,Y_train)
    objective = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader  = DataLoader(dataset,batch_size = 100000)
    for i in range(epochs):
        for j,(X_batch,y_batch) in enumerate(loader):
            opt.zero_grad()
            y_pred = model(X_batch.to(device))
            e = objective(y_pred.squeeze(),y_batch.float().to(device).squeeze()) + reg_parameter*model.lasso_term() #lasso term screwing things up
            e.backward()
            opt.step()
        if i % 1 == 0:
            with torch.no_grad():
                train_auc = auc_check(model, X_train.to(device), Y_train.to(device))
                test_auc = auc_check(model, X_test.to(device), Y_test.to(device))
                print(train_auc)
                print(test_auc)

    return model


