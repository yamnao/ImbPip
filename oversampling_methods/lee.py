from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import itertools

class lee():
    '''
    Reference
    https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_lee.py
    '''
    def __init__(self, nb_data = 1, k_neighbors=5, rejection_level=0.5, n_jobs=None):
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.nb_data = nb_data
        self.rejection_level = rejection_level
    
    def get_parameters_combination(self):
        dico = {'nb_data': [0.2, 0.5, 0.75, 1.0, 1.5, 2.0], 
                'k_neighbors': [3, 5, 7], 
               'rejection_level':[0.2, 0.5, 0.8]}
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
        
        params = min(len(data)-1, self.k_neighbors)
        neigh_all = NearestNeighbors(n_neighbors=params).fit(data)
        dist_all, nns_all = neigh_all.kneighbors(data_min, return_distance=True)
        
        params = min(len(ind_min)-1, self.k_neighbors)
        neigh_min = NearestNeighbors(n_neighbors=params).fit(data_min)
        dist_min, nns_min = neigh_min.kneighbors(data_min, return_distance=True)
        
        samples = []
        passed, trial = 0,0
        rejection_level = self.rejection_level
        while len(samples) < nb_data_to_add:
            if passed == trial and passed > 1000:
                rejection_level += 0.1
                trial, passed = 0, 0
            trial += 1
            ind_select = np.random.choice(range(len(ind_min)))
            ind_neighbor = np.random.choice(nns_min[ind_select][1:])
            step = np.array([random.uniform(0, 1) for _ in range(data.shape[1])])
            sample = data_min[ind_select] + step*(data_min[ind_neighbor] - data_min[ind_select])
            dist_check, ind_check = neigh_all.kneighbors([sample])
            nb_maj_around = np.sum(labels[ind_check][:-1] == class_maj)/params
            if nb_maj_around < rejection_level:
                samples.append(sample)
            else:
                passed += 1
        return np.concatenate((data,samples)), np.concatenate((labels, np.array([class_min]*len(samples))))
        