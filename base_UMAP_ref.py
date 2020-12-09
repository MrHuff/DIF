# TEST COMMIT
import pandas as pd
import umap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
mpl.use("Agg")
import matplotlib.colors as mcolors
colors= list(mcolors.TABLEAU_COLORS.items())

def umap_plot(filename,save_name):
    train_data = pd.read_csv(filename)
    cols = train_data.columns.tolist()
    cols.remove('file_name')
    property_indicator = train_data[cols].values
    reducer_h = umap.UMAP()
    data_umap = reducer_h.fit_transform(property_indicator)
    def subset(input,boolean): #Fix the boolean error...
        boolean = np.array(boolean, dtype=bool)
        subset = input[boolean,:]
        return subset[:,0],subset[:,1]

    fig, ax = plt.subplots()
    for j in range(len(cols)):
        boolean = property_indicator[:, j]
        X, Y = subset(data_umap, boolean)
        print(X[:10], Y[:10])
        print(cols[j])
        ax.scatter(X, Y, label=cols[j], c=colors[j][0], alpha=0.25, marker='.', s=10)
    ax.legend()
    ax.grid(True)
    plt.savefig(f'{save_name}.png')
    plt.clf()

if __name__ == '__main__':
    umap_plot("mnist_full.csv",'mnist_full')
    umap_plot("celebA_hq_gender_multi.csv",'celebA_full')

