import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import shutil
import os



def gen_silhouette_image(n_clusters, X, cluster_labels):
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
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

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.get_cmap('Spectral')(float(i) / n_clusters)
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



    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.savefig('silhouette'+str(n_clusters)+'.png')

n_init = 10
ROOT_DIR = "../"
vgg16_feature_list_pfile = open(ROOT_DIR + 'pickle/vgg16_feature_list.p', 'rb')
vgg16_feature_list = pickle.load(vgg16_feature_list_pfile)
filename_list_pfile = open(ROOT_DIR + 'pickle/filename_list.p', 'rb')
filename_list = pickle.load(filename_list_pfile)

X = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(vgg16_feature_list)



n_components = 30
reducer = TruncatedSVD(n_components=n_components)
reducer.fit(X)
X = reducer.transform(X)
print("%d: Percentage explained: %s\n" % (n_components, reducer.explained_variance_ratio_.sum()))

X = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(X)


k_list = range(2,30)

for n_clusters in k_list:

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters, " the average silhouette_score is :", silhouette_avg)
    print("For n_clusters =", n_clusters, " inertia is :", clusterer.inertia_)


    # Compute the silhouette scores for each sample
    gen_silhouette_image(n_clusters, X, cluster_labels)


    shutil.rmtree(ROOT_DIR + 'clustered_output_' + str(n_clusters),ignore_errors=True)
    os.mkdir(ROOT_DIR + 'clustered_output_' + str(n_clusters))
    for i in range(0,n_clusters):
        os.mkdir(ROOT_DIR + 'clustered_output_' + str(n_clusters) + '/' + str(i))
    for i in zip(filename_list,cluster_labels):
        shutil.copy2(ROOT_DIR + 'data/' + str(i[0]), ROOT_DIR + 'clustered_output_' + str(n_clusters) + '/' + str(i[1]))




# list(map(lambda x:x[0].replace('.jpg',''),(filter(lambda x: x[1]==1, zip(filename_list,cluster_labels)))))
# df = pd.DataFrame(list(zip(filename_list,cluster_labels)))
# df = df.rename(index=str, columns={0: "file", 1: "cluster"})
# df["file"] = df["file"].str.replace(".jpg","")
# df["cluster"] = df["cluster"].apply(lambda x:"Cluster " + str(x+1).zfill(2))
# df['rank'] = df.groupby(['cluster']).cumcount()
# df = pd.pivot_table(df,values=['file'],columns=['cluster'],index=['rank'],aggfunc=np.sum).fillna('')
# df.columns = [col[1] for col in df.columns.values]
# df = df.applymap(lambda x: ("'" + x + "'") if len(x)>0 else x)
# df.to_csv('abc.csv',index=False)