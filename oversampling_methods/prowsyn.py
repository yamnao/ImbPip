from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import itertools

class prowsyn():
    '''
    Reference
    https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_prowsyn.py
    '''
    def __init__(self, nb_data = 1, k_neighbors=5, L=1, theta=0.1, n_jobs=None):
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.nb_data = nb_data
        self.L = L
        self.theta = theta
    
    def get_parameters_combination(self):
        dico = {'nb_data': [0.2, 0.5, 1.0, 2.0], 
                'k_neighbors': [1,3,5], 
               'L':[3,5,10],
               'theta':[0.1,0.5,0.8]}
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

        P = np.array(ind_min.copy())
        nearest_sample, proximity_level = [], []
        for i in range(self.L):
            if len(P) == 0:
                break
            params = min(len(P), self.k_neighbors)
            neigh = NearestNeighbors(n_neighbors=params, n_jobs=self.n_jobs).fit(data[P])
            ind_near_min_maj = np.array(neigh.kneighbors(data_maj, return_distance=False))
            P_i = np.unique(ind_near_min_maj.reshape(-1))
            nearest_sample.append(P[P_i])
            proximity_level.append(np.array([i+1]*len(P_i)))
            P = np.delete(P, P_i, 0)
            
        if len(P) > 0:
            nearest_sample.append(P)
            proximity_level.append(np.array([self.L+1]*len(P)))
        
        proximity_level = np.concatenate(proximity_level, axis=0)
        weights = np.exp(-self.theta*(np.array(proximity_level)-1))
        weights = weights/np.sum(weights, axis=0)
        weights = weights*nb_data_to_add
        
        samples = []
        count = 0
        for i in range(len(nearest_sample)):
            for j in range(len(nearest_sample[i])):
                if int(weights[count+j]) > 1:
                    nb_to_add =int(weights[count+j])
                else:
                    nb_to_add = 1
                ind_select = np.array(random.choices(nearest_sample[i],
                                      k=nb_to_add))
                
                steps = np.array([[random.uniform(0, 1) for _ in range(data.shape[1])] for e in range(len(ind_select))])
                
                samples.append(data[nearest_sample[i][j]] + steps * (data[ind_select] - data[nearest_sample[i][j]] ))
            count += len(nearest_sample[i])

        samples = np.concatenate(samples, axis=0)
        
        return np.concatenate((data,samples)), np.concatenate((labels, np.array([class_min]*len(samples))))
       
        
            
            