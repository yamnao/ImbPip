from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import itertools

class smote():
    '''
    Reference:
    https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_smote.py
    '''
    def __init__(self, nb_data = 1, k_neighbors=5,n_jobs=None):
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.nb_data = nb_data
    
    def get_parameters_combination(self):
        dico = {'nb_data': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                'k_neighbors': [3, 5, 7]}
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

        neigh = NearestNeighbors(n_neighbors=self.k_neighbors+1).fit(data_min)
        nns = np.array(neigh.kneighbors(data_min, return_distance=False)[:, 1:])
        nns = nns.reshape(-1)
        ind_pt = np.array([[e for _ in range(self.k_neighbors)] for e in np.array(ind_min)])
        ind_pt = ind_pt.reshape(-1)
        
        ind_select = np.array(random.choices(range(len(ind_pt)), k=nb_data_to_add))
        
        steps = np.array([[random.uniform(0, 1) for _ in range(data.shape[1])] for e in range(nb_data_to_add)])

        samples = data_min[nns[ind_select]] + steps * (data[ind_pt[ind_select]] - data_min[nns[ind_select]] )

        return np.concatenate((data,samples)), np.concatenate((labels, np.array([class_min]*len(samples))))