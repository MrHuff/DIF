import umap
import numpy as np
import torch
import matplotlib.pyplot as plt
def reject_outliers(data, m=2):
    norm = np.linalg.norm(data,axis=1)
    return data[abs(norm - np.mean(norm)) < m * np.std(norm),:]

def make_binary_class_umap_plot(all_h,c,save_path,cur_it,description):
    dataA_np = all_h[~c,:]
    dataB_np = all_h[c,:]
    reducer_h = umap.UMAP()
    reducer_h.fit(all_h)

    A_data_umap = reducer_h.transform(dataA_np)
    B_data_umap = reducer_h.transform(dataB_np)

    A_umap_no_outlier = reject_outliers(A_data_umap)
    B_umap_no_outlier = reject_outliers(B_data_umap)
    fig, ax = plt.subplots()
    ax.scatter(A_umap_no_outlier[:, 0], A_umap_no_outlier[:, 1], label='A', c='red', alpha=0.5, marker='.', s=10)
    ax.scatter(B_umap_no_outlier[:, 0], B_umap_no_outlier[:, 1], label='B', c='blue', alpha=0.5, marker='.', s=10)
    ax.legend()
    ax.grid(True)
    plt.savefig(f'{save_path}/intro-vae_umap_{description}_{cur_it}.png')



