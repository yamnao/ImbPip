from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import itertools
from sklearn.cluster import OPTICS

class smobd():
    '''
    Reference:
    https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_smobd.py
    '''
    def __init__(self, nb_data = 1, k_neighbors=5, noisy_threshold=0.1, eps=5, eta1= 0.2, n_jobs=None):
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.noisy_threshold = noisy_threshold
        self.nb_data = nb_data
        self.eps = eps
        self.eta1 = eta1    
    
    def get_parameters_combination(self):
        dico = {'nb_data': [0.2, 0.5, 1.0, 2.0], 
                'k_neighbors': [3, 5, 7], 
               'noisy_threshold':[0.1, 0.5,1], 
               'eps':[1, 5, 10], 
               'eta1':[0.1,0.5,1]}
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
        optic = OPTICS(min_samples=params, n_jobs=self.n_jobs).fit(data_min)
        core_distance = optic.core_distances_
        reach_distance = optic.reachability_
       
        noise = np.logical_and(core_distance > np.mean(core_distance)*self.noisy_threshold, reach_distance > np.mean(reach_distance)*self.noisy_threshold)
        count_noise = np.sum(noise, axis=0)
        
        neigh = NearestNeighbors(n_neighbors=self.k_neighbors+1).fit(data_min)
        nns = np.array(neigh.kneighbors(data_min, return_distance=False))
        radius = np.array([len(x) for x in neigh.radius_neighbors(
            data_min, radius=self.eps, return_distance=False)])
        
        df = core_distance*self.eta1 + radius*(1-self.eta1)
        df[noise==True] = 0
        df = df/np.sum(df, axis=0)
        
        samples = []
        for _ in range(nb_data_to_add):
            ind_select = np.array(random.choices(range(len(ind_min)),k=1, weights=df))
            neighbor_idx = np.array(random.choices(nns[ind_select][0][1:], k=1))
            steps = np.array([random.uniform(0, 1) for _ in range(data.shape[1])])
            sample = data[ind_min[ind_select]] + steps * (data[neighbor_idx] - data[ind_min[ind_select]])
            samples.append(sample[0])
        
        return np.concatenate((data,samples)), np.concatenate((labels, np.array([class_min]*len(samples))))
       