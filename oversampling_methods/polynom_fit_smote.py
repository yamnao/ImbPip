import numpy as np
import random
import itertools

class polynom_fit_smote():
    '''
    Reference:
    https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_polynom_fit_smote.py
    '''
    def __init__(self, nb_add=1, interpolation='star'):
        self.nb_add = nb_add
        self.interpolation = interpolation
        
    def get_parameters_combination(self):
        dico = {'nb_add': [0.2, 0.5, 1.0, 1.5, 2.0], 
                'interpolation': ['star', 'mesh', 'bus', 'poly_2', 'poly_3']}
        keys, values = zip(*dico.items())
        return np.array([dict(zip(keys, v)) for v in itertools.product(*values)])
    
    def interpolation_star(self, data, labels):
        X_mean = np.mean(self.data_min, axis=0)
        k = max([1, int(np.rint(self.nb_data_to_add/len(self.data_min)))])
        samples = []
        for x in self.data_min:
            diff = X_mean - x
            for i in range(1, k+1):
                samples.append(x + float(i)/(k+1)*diff)
        return np.concatenate((data, samples)), np.concatenate((labels, [self.class_min]*len(samples)))
    
    def interpolation_bus(self, data, labels):
        k = max([1, int(np.rint(self.nb_data_to_add/len(self.data_min)))])
        samples = []
        for i in range(1, len(self.data_min)):
            diff = self.data_min[i-1] - self.data_min[i]
            for j in range(1, k+1):
                samples.append(self.data_min[i] + float(j)/(k+1)*diff)
        return np.concatenate((data, samples)), np.concatenate((labels, [self.class_min]*len(samples)))
    
    
    def interpolation_mesh(self, data, labels):
        samples = []
        if len(self.data_min)**2 > self.nb_data_to_add:
            while len(samples) < self.nb_data_to_add:
                random_i = random.randrange(0,len(self.data_min))
                random_j = random.randrange(0, len(self.data_min))
                diff = self.data_min[random_i] - self.data_min[random_j]
                samples.append(self.data_min[random_i] + 0.5*diff)
        else:
            n_combs = (len(self.data_min)*(len(self.data_min)-1)/2)
            k = max([1, int(np.rint(self.nb_data_to_add/n_combs))])
            for i in range(len(self.data_min)):
                for j in range(len(self.data_min)):
                    diff = self.data_min[i] - self.data_min[j]
                    for li in range(1, k+1):
                        samples.append(self.data_min[j] + float(li)/(k+1)*diff)
        return np.concatenate((data, samples)), np.concatenate((labels, [self.class_min]*len(samples)))
    
    
    def fit_poly(self, d):
        return np.poly1d(np.polyfit(np.arange(len(self.data_min)), self.data_min[:, d], self.deg))

    def interpolation_poly(self, data, labels):
        dim = len(self.data_min[0])
        polys = [self.fit_poly(d,) for d in range(dim)]
        samples = []
        for d in range(dim):
            for _ in range(self.nb_data_to_add):
                random_i = random.randrange(0,len(self.data_min))
                samples_gen = [polys[d](el)
                               for el in self.data_min[random_i]]
                samples.append(np.array(samples_gen).T)
        return np.concatenate((data, samples)), np.concatenate((labels, [self.class_min]*len(samples)))
    
    def fit_sample(self, data, labels):
        classes, count_class = np.unique(labels, return_counts = True)
        self.class_min = classes[0] if count_class[0] < count_class[1] else classes[1]
        self.class_maj = classes[0] if count_class[0] > count_class[1] else classes[1]
        
        self.nb_data_to_add = int((count_class[self.class_maj] - count_class[self.class_min])*self.nb_add)
        
        ind_min = np.where(labels == self.class_min)[0]
        self.data_min = data[labels == self.class_min]
        self.data_maj = data[labels == self.class_maj]
        
        if self.interpolation == 'star':
            return self.interpolation_star(data, labels)
        elif self.interpolation == 'poly_2':
            self.deg = 2
            return self.interpolation_poly(data, labels)
        elif self.interpolation == 'poly_3':
            self.deg = 3
            return self.interpolation_poly(data, labels)
        elif self.interpolation == 'mesh':
            return self.interpolation_mesh(data, labels)
        elif self.interpolation == 'bus':
            return self.interpolation_bus(data, labels)
        