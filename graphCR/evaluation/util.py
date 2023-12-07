import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def scatter(x, colors, title):
    # choose a color palette with seaborn.
    # pca = PCA(n_components=2)
    result = TSNE(init="pca", learning_rate='auto', random_state=42).fit_transform(x)
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.set_title(title)
    sc = ax.scatter(result[:, 0], result[:, 1], lw=0, s=40, color=colors)
    plt.title = title
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []
    plt.show()
    return f, ax, sc, txts

def scatter_pca(x, colors, title):
    # choose a color palette with seaborn.
    pca = PCA(n_components=2)
    #result = TSNE(random_state=42).fit_transform(x)
    result = pca.fit_transform(x)
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.set_title(title, pad = -10)
    sc = ax.scatter(result[:, 0], result[:, 1], lw=0, s=40, color=colors)
    plt.title = title
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []
    plt.show()
    return f, ax, sc, txts