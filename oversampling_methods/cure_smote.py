from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
import numpy as np
import random
import itertools

class cure_smote():
    '''
    Reference
    https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_cure_smote.py
    '''
    def __init__(self, nb_data = 1, nb_clust=5, noise_th=2, n_jobs=None):
        self.nb_clust = nb_clust
        self.noise_th = noise_th
        self.n_jobs = n_jobs
        self.nb_data = nb_data
    
    def get_parameters_combination(self):
        dico = {'nb_data': [0.1, 0.25, 0.5, 1.0, 1.5, 2.0], 
                'nb_clust': [5, 10, 15],
               'noise_th':[1,2,3]}
        keys, values = zip(*dico.items())
        return np.array([dict(zip(keys, v)) for v in itertools.product(*values)])
        
        
    def fit_sample(self, data, labels):
        classes, count_class = np.unique(labels, return_counts = True)
        class_min = classes[0] if count_class[0] < count_class[1] else classes[1]
        class_maj = classes[0] if count_class[0] > count_class[1] else classes[1]

        ind_min = np.where(labels == class_min)[0]
        data_min = data[labels == class_min]
        data_maj = data[labels == class_maj]

        nb_data_to_add = int((len(data_maj) - len(data_min))*self.nb_data)
        
        data_ = MinMaxScaler().fit_transform(data)
        data_min_ = data_[labels == class_min]
        
        distances_between_data = pairwise_distances(data_min_)
        for i in range(len(distances_between_data)):
            distances_between_data[i, i] = np.inf
            
        clusters = [np.array([i]) for i in range(len(data_min_))]
        
        iteration_growth = 0
        while len(clusters) > self.nb_clust:
            iteration_growth += 1
            if iteration_growth % 10 == 0:
                sizes = np.array([len(c) for c in clusters])
                to_remove = np.where(sizes == np.min(sizes))[0]
                to_remove = np.random.choice(to_remove)
                del clusters[to_remove]
                distances_between_data = np.delete(distances_between_data, to_remove, axis=0)
                distances_between_data = np.delete(distances_between_data, to_remove, axis=1)
            
            min_coord = np.where(distances_between_data == np.min(distances_between_data))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]
            
            clusters[merge_a] = np.hstack([clusters[merge_a], clusters[merge_b]])
            del clusters[merge_b]
            distances_between_data[merge_a] = np.min(np.vstack([distances_between_data[merge_a], distances_between_data[merge_b]]), axis=0)
            distances_between_data[:, merge_a] = distances_between_data[merge_a]
            distances_between_data = np.delete(distances_between_data, merge_b, axis=0)
            distances_between_data = np.delete(distances_between_data, merge_b, axis=1)
            for i in range(len(distances_between_data)):
                distances_between_data[i, i] = np.inf

        to_remove = []
        for i in range(len(clusters)):
            if len(clusters[i]) < self.noise_th:
                to_remove.append(i)
        clusters = [clusters[i]
                    for i in range(len(clusters)) if i not in to_remove]

        if len(clusters) == 0:
            return data.copy(), labels.copy()

        samples = []
        for _ in range(nb_data_to_add):
            cluster_idx = np.random.choice(len(clusters))
            center = np.mean(data_min_[clusters[clusters==cluster_idx]], axis=0)
            representative = data_min_[np.random.choice(clusters[clusters==cluster_idx])]
            step = np.array([random.uniform(0, 1) for _ in range(data.shape[1])])
            samples.append(center + step*(representative - center))

        return np.concatenate((data_,samples)), np.concatenate((labels, np.array([class_min]*len(samples))))
            
            