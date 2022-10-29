from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from oversampling_methods.smote import smote
import numpy as np
import random
import itertools

class smote_ipf():
    '''
    Reference:
    https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_smote_ipf.py
    '''
    def __init__(self, nb_data = 1, k_neighbors=5,voting_scheme='majority', classifier= DecisionTreeClassifier(random_state=2), n_folds=9, n_jobs=None):
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.nb_data = nb_data
        self.voting_scheme = voting_scheme
        self.n_folds = n_folds
        self.classifier = classifier
    
    def get_parameters_combination(self):
        dico = {'nb_data': [0.2, 0.5, 0.75, 1.0, 1.5, 2.0], 
                'k_neighbors': [3, 5, 7],
               'voting_scheme':['consensus', 'majority'], 
               'n_folds':[9], 
               'classifier':[DecisionTreeClassifier(random_state=2)]}
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
        
        data_over, labels_over = smote(nb_data = self.nb_data, 
                                       k_neighbors=self.k_neighbors).fit_sample(data, labels)
        
        
        n_folds = min(self.n_folds, len(ind_min)-1)
        
        nb_iteration = 0
        while True:
            predictions = []
            for train_index, _ in StratifiedKFold(n_folds).split(data_over, labels_over):
                self.classifier.fit(data_over[train_index], labels_over[train_index])
                predictions.append(self.classifier.predict(data_over))
                
            pred_votes = (np.mean(predictions, axis=0) > 0.5).astype(int)
            if self.voting_scheme == 'majority':
                to_remove = np.where(np.not_equal(pred_votes, labels_over))[0]
            elif self.voting_scheme == 'consensus':
                sum_votes = np.sum(predictions, axis=0)
                to_remove = np.where(np.logical_and(np.not_equal(
                    pred_votes, labels_over), np.equal(sum_votes, self.n_folds)))[0]
            
            data_over = np.delete(data_over, to_remove, axis=0)
            labels_over = np.delete(labels_over, to_remove)
            if len(to_remove) < len(data_over)*0.01:
                nb_iteration +=1
            else:
                nb_iteration = 0
            if nb_iteration >= 3:
                break
        return data_over, labels_over