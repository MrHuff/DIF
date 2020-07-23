import pandas as pd
from post_processing_script import save_paths_faces,save_paths_fashion,save_paths_mnist

if __name__ == '__main__':
    concat = []
    names=['CelebHQ-DIF','CelebHQ-Vanilla','CelebHQ-linear']
    for i,el in enumerate(save_paths_faces):
        df = pd.read_csv(el+'/summary.csv',index_col=0)
        cols = df.columns.tolist()
        new_row = [names[i]]
        for c in cols:
            mean = round(df[c]['mean'],2)
            std = round(df[c]['std'],2)
            new_row.append(str(mean)+'$\pm$'+str(std))
        concat.append(new_row)
    cols = ['dataset-model']+cols
    cols = [el.replace('_','-') for el in cols]
    new_df = pd.DataFrame(concat,columns=cols)
    print(new_df.to_latex(index=False,escape=False))




