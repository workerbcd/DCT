import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import manifold
import os

def draw_tsne(features,labels, path,evalnum,name=None,prototypes=None):
    if not os.path.exists(path):
        os.makedirs(path)
    savepath = os.path.join(path,"eval{}_show_tsne.jpg".format(evalnum))
    x = features.cpu().detach().numpy()
    y = labels.cpu().numpy()
    '''t-sne'''
    tsne = manifold.TSNE(n_components=2,init='pca',random_state=501)
    X_tsne = tsne.fit_transform(x)
    print("original dimention is {}. Embedding dimention is {}".format(x.shape[-1],X_tsne.shape[-1]))

    '''Visualization'''
    x_min,x_max = X_tsne.min(0),X_tsne.max(0)
    x_norm = (X_tsne-x_min)/(x_max-x_min)

    plt.figure(figsize=(16,16))
    # colors = [plt.cm.Set1(i) for i in range(len(np.unique(y)))]

    for i in range(x_norm.shape[0]):
        if y[i]>8: continue
        plt.text(x_norm[i,0],x_norm[i,1],'*',color=plt.cm.Set1(y[i]))
    # for i in range(x_norm.shape[0]-len(name),x_norm.shape[0]):
    #     plt.text(x_norm[i, 0], x_norm[i, 1], str(y[i]), fontsize=15,fontweight='bold', color='black')
        # for i in range(x_norm.shape[0]):
        #     plt.text(x_norm[i,0],x_norm[i,1],'*',color=plt.cm.Set1(y[i]),label=name[y[i]])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='best')
    plt.savefig(savepath)
    plt.close()
