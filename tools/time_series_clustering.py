import numpy as np
import pandas as pd
import copy
import ruptures as rpt
import scipy.stats as sts
from sklearn.cluster import KMeans, AgglomerativeClustering
from tools.operators import autocorr, chi
from scipy.stats import kstest

def cpd(ts: np.array, kernel='l2', jump=2, min_size=128, penalty=1):

    algo = rpt.Pelt(model=kernel, jump=jump, min_size=min_size).fit(ts)
    result = algo.predict(pen=penalty)

    return result

def ts_regimes(ts: np.array, kernel='l2', jump=2, min_size=128, penalty=1):

    idx = cpd(ts, kernel, jump, min_size, penalty)

    switch_regimes = copy.deepcopy(idx)
#     switch_regimes.append(len(ts)) #rupture include last element otherwise use this
    switch_regimes.insert(0, 0)

    array = []

    for i in range(len(switch_regimes) - 1):
        array.append(ts[switch_regimes[i]:switch_regimes[i+1]])

    return np.array(array), switch_regimes

def splitted_plot(splitted, idx): #visualization of splitted ts
    plt.subplots(figsize=(10, 5))

    for i in range(len(idx) - 1):
        plt.plot(np.arange(idx[i], idx[i+1]), splitted[i], c='black')

    return plt.show()


def split_ts_clustering(ts: np.array, num_clusters=3, kernel='l2', jump=10, min_size=100, penalty=10):
    splitted, idx = ts_regimes(ts, kernel, jump, min_size, penalty)
    train = np.array(list(map(chi, splitted)))

    # cluster_model = KMeans(n_clusters=num_clusters, random_state = 5)
    cluster_model = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward', affinity='euclidean')
    cluster_model.fit(train)

    clusters = cluster_model.labels_

    #Take clusters' indeces
    index_array = []
    for i in range(num_clusters):
        index_array.append(np.where(clusters==i)[0])

    #split time series by choosen indeces
    data_clustered = []
    for i in range(num_clusters):
        data_clustered.append(splitted[index_array[i]])

    #stack choosen data by clusters
    stacked_clusters = []
    length_clusters = []
    delta_array = []
    for i in range(num_clusters):
        deltas = []
        for k in range(len(data_clustered[i]) - 1):
            delta = data_clustered[i][k][-1] - data_clustered[i][k+1][0]

            if delta >= 0:
                data_clustered[i][k+1] = data_clustered[i][k+1] + delta + 1e-6

            else:
                data_clustered[i][k+1] = data_clustered[i][k+1] - abs(delta) + 1e-6

            deltas.append(delta)

        delta_array.append(deltas)
        stacked_clusters.append(np.hstack(data_clustered[i]))
        length_clusters.append(list(map(len , data_clustered[i])))

    stacked_clusters = np.array(stacked_clusters)
    length_clusters = np.array(length_clusters)

    return stacked_clusters, length_clusters, index_array, idx, splitted, delta_array

def ts_reconstruction(synth_ts, data, length_clusters, num_clusters, index_array, delta_array):
    ts_window_rec = []

    for j in range(num_clusters):

        ts_window_rec_cluster = []
        k = 0
        for i in length_clusters[j]:
            ts_window_rec_cluster.append(synth_ts[j][k:k+i])
            k += i

        ts_window_rec_cluster = np.array(ts_window_rec_cluster)
        ts_window_rec.append(ts_window_rec_cluster)

    ts_window_rec = np.array(ts_window_rec)

    #add deltas
    for j in range(num_clusters):
        for i in range(1, len(ts_window_rec[j])):
            ts_window_rec[j][i] += -delta_array[j][i-1] - 1e-6

    #reconstruct indeces

    rec_stack = []
    for i in range(len(ts_window_rec)):
        for j in range(len(ts_window_rec[i])):
            rec_stack.append(ts_window_rec[i][j])

    zipped = list(zip(rec_stack, np.hstack(index_array))) #beacuse np.hstack works not good sometimes with 1-d array
    # zipped = list(zip(np.hstack(ts_window_rec), np.hstack(index_array)))
    sorted_array = np.array(sorted(zipped, key = lambda x: x[1]))
    result = sorted_array[:, 0]

    result = np.hstack(result)
    return result
