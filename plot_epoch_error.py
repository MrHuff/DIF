import tensorflow.compat.v1 as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as  plt
sns.set_style("whitegrid")
tf.disable_v2_behavior()

def get_df(file_path):
    cols = ['step']
    data = []
    row_counter = 0
    step = -10
    for i,e in enumerate(tf.train.summary_iterator(file_path)):
        if step!=e.step:
            if row_counter>0:
                data.append(row)
            step = e.step
            row_counter+=1
            row = [e.step]
        for v in e.summary.value:
            row.append(v.simple_value)
            if row_counter==2:
                cols.append(v.tag)
    return pd.DataFrame(data[1:],columns=cols)

if __name__ == '__main__':
    files=[
        'resultsmnist38_epochs_bs=32_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.01_kernel=linear_tanh=True_C=10.0_linearb=False_J=0.25/events.out.tfevents.1655464654.zizgpu04.cpu.stats.ox.ac.uk',
        'resultsfacesHQv3_epochs_bs=32_beta=1.0_KL=1.0_KLneg=0.5_fd=3_m=1000.0_lambda_me=0.25_kernel=linear_tanh=True_C=10.0_linearb=False_J=0.25/events.out.tfevents.1655405997.zizgpu04.cpu.stats.ox.ac.uk',
        'resultscovid256_epochs_bs=32_beta=0.25_KL=1.0_KLneg=0.5_fd=3_m=150.0_lambda_me=0.15_kernel=linear_tanh=True_C=10.0_linearb=False_J=0.25/events.out.tfevents.1655466036.zizgpu04.cpu.stats.ox.ac.uk'
    ]
    col_name = ['loss_rec','lossE_real_kl','lossE_rec_kl','lossE_fake_kl','lossG_rec_kl','lossG_fake_kl']
    col_name_new = ['Reconstruction Loss',r'$KL^{real}_{E}$',r'$KL^{rec}_{E}$',r'$KL^{fake}_{E}$',r'$KL^{rec}_{G}$',r'$KL^{fake}_G$']
    for name,f in zip(['MNIST','CelebA','COVID'],files):
        df = get_df(f)
        df['step']=df['step']-df['step'].min()
        df=df.iloc[::5, :]
        df = df.rename({a:b for a,b in zip(col_name,col_name_new)}, axis=1)  # new method
        tmp = pd.melt(df,id_vars=['step'],value_vars=col_name_new)
        plt.figure(figsize=(32, 8))
        b=sns.lineplot(data=tmp, x="step", y='value',hue="variable",alpha=0.5)
        plt.legend(fontsize=20,loc='upper right')
        plt.xlabel('Iteration', fontsize=25)
        plt.ylabel('Value', fontsize=25)
        b.set_yticklabels(b.get_yticks(), size=15)
        b.set_xticklabels(b.get_xticks(), size=15)
        plt.savefig(f'train_loss_{name}.png',bbox_inches='tight',
               pad_inches=0)



