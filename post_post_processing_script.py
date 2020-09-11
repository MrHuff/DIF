import pandas as pd
from post_processing_script import save_paths_faces,save_paths_fashion,save_paths_mnist,save_paths_covid

dataset = [save_paths_faces,save_paths_fashion,save_paths_mnist,save_paths_covid]
cnn_ref = ['celeb_classify/','fashion_classify/','mnist_classify/','covid_classify/']
if __name__ == '__main__':
    concat = []
    names=[['CelebHQ-DIF','CelebHQ-Vanilla','CelebHQ-linear'],['Fashion-DIF','Fashion-Vanilla','Fashion-linear'],['MNIST-DIF','MNIST-Vanilla','MNIST-linear'],['Covid-DIF','Covid-Vanilla','Covid-linear']]
    sp = 3
    for i,el in enumerate(dataset[sp]):
        ref_df = pd.read_csv(el+'/summary.csv',index_col=0)
        cnn_df = pd.read_csv(cnn_ref[sp] + 'performance_summary.csv',index_col=0)
        df = pd.concat([ref_df,cnn_df['test auc']],axis=1)
        cols = df.columns.tolist()
        new_row = [names[sp][i]]
        for c in cols:
            mean = round(df[c]['mean'],3)
            std = round(df[c]['std'],3)
            if 'sparsity' in c:
                new_row.append(str(round(mean*100,1))+'\%'+'$\pm$'+str(round(std*100,1  ))+'\%')
            else:
                new_row.append(str(mean)+'$\pm$'+str(std))
        concat.append(new_row)
    cols = ['dataset-model']+cols
    cols = [el.replace('_','-') for el in cols]
    new_df = pd.DataFrame(concat,columns=cols)
    # lasso_cols = ['test-auc-0','test-auc-0.001','test-auc-0.01','test-auc-0.1','test-auc-1.0','test auc']
    # lasso_df = new_df[['dataset-model']+lasso_cols]
    # new_df = new_df.drop(lasso_cols,axis=1)
    # lasso_df.columns= ['dataset-model','$\lambda_{\text{lasso}}=0$','$\lambda_{\text{lasso}}=0.001$','$\lambda_{\text{lasso}}=0.01$','$\lambda_{\text{lasso}}=0.1$','$\lambda_{\text{lasso}}=1.0$','CNN']
    # print(lasso_df)
    # print(new_df)


    print(new_df.to_latex(index=False,escape=False))
    # print(lasso_df.to_latex(index=False,escape=False))
    # sub_df = new_df[['dataset-model','fake-FID','prototype-FID','ELBO','log-likelihood']]
    # sub_df_2 = lasso_df.iloc[:,1:]
    # concat_df = pd.concat([sub_df,sub_df_2],axis=1)
    # print(concat_df.to_latex(index=False,escape=False))






