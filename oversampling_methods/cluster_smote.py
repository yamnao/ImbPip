from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
import random 
import itertools

class cluster_smote():
    '''
    Reference:
    https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_cluster_smote.py
    '''
    
    def __init__(self, n_neighbors=2, n_clusters=10):
        self.n_neighbors = n_neighbors 
        self.n_clusters = n_clusters
    
    def get_parameters_combination(self):
        dico = {
            'n_neighbors':[2, 3, 5, 10],
            'n_clusters':[3, 10, 20, 50, 100, 200]
        }
        keys, values = zip(*dico.items())
        return np.array([dict(zip(keys, v)) for v in itertools.product(*values)])
    
    def fit_sample(self, train, train_labels):
        
        classes, count_class = np.unique(train_labels, return_counts = True)
        class_min = classes[0] if count_class[0] < count_class[1] else classes[1]
        class_maj = classes[0] if count_class[0] > count_class[1] else classes[1]
        ind_min = np.where(train_labels == class_min)[0]
        train_min = train[train_labels == class_min]
        train_maj = train[train_labels == class_maj]
        
        kmeans = KMeans(n_clusters = min([len(train_min), self.n_clusters]))
        kmeans.fit(train_min)
        
        samples = []
        for clust in np.unique(kmeans.labels_):
            ind_labels = np.array(np.where(kmeans.labels_ == clust)[0])
            if len(ind_labels) > 2:
                nn = NearestNeighbors(n_neighbors = min(len(ind_labels)-1, self.n_neighbors))
                nn.fit(train_min[ind_labels])
                kneighbors = nn.kneighbors()[1]

                steps = np.array([[random.uniform(0, 1) for _ in range(train.shape[1])] for e in range(len(kneighbors[:,0]))])

                samples.append(train_min[ind_labels][np.array(kneighbors[:,0])] + steps * (train_min[ind_labels][np.array(kneighbors[:,1])] - train_min[ind_labels][np.array(kneighbors[:,0])]))
        
        if samples != []:
            samples = np.concatenate(samples, axis=0 )
            return np.concatenate((train, samples)), np.concatenate((train_labels, [class_min]*len(samples)))
        
        return train, train_labels
        
        
        