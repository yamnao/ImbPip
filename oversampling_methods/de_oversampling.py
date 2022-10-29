from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import itertools
from sklearn.cluster import KMeans


class de_oversampling():
    '''
    Reference:
    https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_de_oversampling.py
    '''
    def __init__(self, nb_data = 1, k_neighbors=5,nb_clust =50, cross_over=0.5, similarity = 0.5, n_jobs=None):
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.nb_data = nb_data
        self.nb_clust = nb_clust
        self.similarity = similarity
        self.cross_over = cross_over
    
    def get_parameters_combination(self):
        dico = {'nb_data': [0.2, 0.5, 1.0, 2.0], 
                'k_neighbors': [3, 5, 10],
               'nb_clust':[20, 50, 100], 
               'cross_over':[0.2, 0.5, 0.8], 
               'similarity':[0.2,0.5]}
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
        
        params = min(len(ind_min)-1, self.k_neighbors)
        
        neigh = NearestNeighbors(n_neighbors=params).fit(data_min)
        nns = neigh.kneighbors(data_min, return_distance=False)
        
        samples = []
        for _ in range(nb_data_to_add):
            index_select = random.choice(range(len(ind_min)))
            index_neighbor = np.random.choice(nns[index_select][1:],2, replace=False)
            step = np.array([random.uniform(0, 1) for _ in range(data.shape[1])])
            mutation = data[ind_min[index_select]] + (data[ind_min[index_neighbor[0]]] - data[ind_min[index_neighbor[1]]])*step 
            rand_s = random.choice(range(data.shape[1]))
            for j in range(data.shape[1]):
                if j != rand_s and random.uniform(0, 1) > self.cross_over:
                    mutation[j] = data[ind_min[index_select]][j]
            samples.append(mutation)
        
        data, labels = np.concatenate((data,samples)), np.concatenate((labels, np.array([class_min]*len(samples))))
        
        params = min(len(data)-1, self.nb_clust)
        kmeans = KMeans(n_clusters=self.nb_clust).fit(data)
        labels_clust = kmeans.labels_
        labels_keep = [label_clust for label_clust in np.unique(labels_clust) if len(np.unique(labels[labels_clust == label_clust])) == class_min]
           
        to_remove = []
        for l in labels_keep:
            ind_l = np.array(np.where(np.array(labels_clust) == l)[0])
            if len(ind_l) >= 3:
                data_clust = data[ind_l]
                data_mean = np.mean(data_clust, axis=0)
                neigh = NearestNeighbors(n_neighbors=2).fit(data_clust)
                nns = neigh.kneighbors([data_mean], return_distance=False)[:, 1:]
                distance = np.linalg.norm(data[nns] - data_mean)
                if distance > self.similarity:
                    to_remove.append(nns)
        
        return np.delete(data, to_remove, 0), np.delete(labels, to_remove, 0)
        