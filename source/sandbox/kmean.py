# Sebastian Raschka, 2015 (http://sebastianraschka.com)
# Python Machine Learning - Code Examples
#
# Chapter 6 - Learning Best Practices for Model Evaluation
#             and Hyperparameter Tuning
#
# S. Raschka. Python Machine Learning. Packt Publishing Ltd., 2015.
# GitHub Repo: https://github.com/rasbt/python-machine-learning-book
#
# License: MIT
# https://github.com/rasbt/python-machine-learning-book/blob/master/LICENSE.txt


# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version

import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn import cluster
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

import sklearn
sklearn.__version__

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans


#############################################################################
print(50 * '=')
print('Section: Loading the Breast Cancer Wisconsin dataset')
print(50 * '-')

df = pd.read_csv('wdbc.data', header=None)
print('Breast Cancer dataset excerpt:\n\n')
print(df.head())

print('Breast Cancer dataset dimensions:\n\n')
print(df.shape)

X = df.loc[:, 2:].values
# y = df.loc[:, 1].values
# le = LabelEncoder()
# y_train = le.fit_transform(y)
# y_enc = le.transform(['M', 'B'])
# print("Label encoding example, le.transform(['M', 'B'])")
# print(le.transform(['M', 'B']))

lsa = make_pipeline(StandardScaler())

X = lsa.fit_transform(X)

print('mean: ',np.mean(X,axis=0))
print('SD: ',np.std(X,axis=0))

k_list = range(2,11)

for k in k_list:
    k_means = cluster.KMeans(n_clusters=k)
    k_means.fit(X)
    print('k=',k,'/ inertia=',k_means.inertia_)


random_state = np.random.RandomState(0)

# Number of run (with randomly generated dataset) for each strategy so as
# to be able to compute an estimate of the standard deviation
n_runs = 1
n_init = 20
# k-means models can do several random inits so as to be able to trade
# CPU time for convergence robustness
# n_init_range = np.array([1, 5, 10, 15, 20])



# Part 1: Quantitative evaluation of various init methods

fig = plt.figure()
plots = []
legends = []

cases = [
    (KMeans, 'k-means++', {}),
    (KMeans, 'random', {}),
   # (MiniBatchKMeans, 'k-means++', {'max_no_improvement': 3}),
   # (MiniBatchKMeans, 'random', {'max_no_improvement': 3, 'init_size': 500}),
]

for factory, init, params in cases:
    print("Evaluation of %s with %s init" % (factory.__name__, init))
    inertia = np.empty((len(k_list), n_runs))

    for run_id in range(n_runs):
        for i,k in enumerate(k_list):
            km = factory(n_clusters=k, init=init, random_state=run_id, n_init=n_init, **params).fit(X)
            inertia[i, run_id] = km.inertia_
            print("inertia[%s,%s] = %s" % (i,run_id,inertia[i,run_id]))
            labels = km.labels_
            print(sklearn.metrics.silhouette_score(X, labels, metric='euclidean'))
    p = plt.errorbar(k_list, inertia.mean(axis=1), inertia.std(axis=1))
    plots.append(p[0])
    legends.append("%s with %s init" % (factory.__name__, init))



plt.xlabel('k')
plt.ylabel('inertia')
plt.legend(plots, legends)
plt.title("Mean inertia for various k-means init across %d runs" % n_runs)

plt.show()



for n_clusters in k_list:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=run_id, n_init=n_init)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # # 2nd Plot showing the actual clusters formed
    # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #             c=colors, edgecolor='k')
    #
    # # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # # Draw white circles at cluster centers
    # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #             c="white", alpha=1, s=200, edgecolor='k')
    #
    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                 s=50, edgecolor='k')
    #
    # ax2.set_title("The visualization of the clustered data.")
    # ax2.set_xlabel("Feature space for the 1st feature")
    # ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.savefig('silhouette'+str(n_clusters)+'.png')