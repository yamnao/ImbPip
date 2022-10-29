import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import warnings
import random
import itertools

class assembled_smote():
    '''
    Reference: https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_assembled_smote.py
    '''
    
    def __init__(self, proportion=1, n_neighbors=5, pop=3, thres=0.3):
        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.pop = pop
        self.thres = thres
        
    def get_parameters_combination(self):
        dico = {
            'proportion':[0.2, 0.5, 0.7, 1, 2],
            'n_neighbors':[3, 5, 10],
            'pop':[2, 5],
            'thres':[0.1,0.3,0.5]
        }
        keys, values = zip(*dico.items())
        return np.array([dict(zip(keys, v)) for v in itertools.product(*values)])
    
    def determine_border_non_border(self, train, labels, train_min):

        # fitting nearest neighbors model
        n_neighbors = min([len(train), self.n_neighbors+1])
        nearestn= NearestNeighbors(n_neighbors=n_neighbors)
        nearestn.fit(train )
        ind = nearestn.kneighbors(train_min, return_distance=False)

        # finding the set of border and non-border minority elements
        n_min_neighbors = [np.sum(labels[ind[i]] == self.class_min)
                           for i in range(len(ind))]
        border_mask = np.logical_not(np.array(n_min_neighbors) == n_neighbors)
        train_border = train_min[border_mask] # pylint: disable=invalid-name
        train_non_border = train_min[np.logical_not(border_mask)] # pylint: disable=invalid-name

        return train_border, train_non_border

    def do_the_clustering(self, train_border):
        
        clusters = [np.array([i]) for i in range(len(train_border))]

        distm = pairwise_distances(train_border)
        for idx in range(len(distm)):
            distm[idx, idx] = np.inf

        # do the clustering
        while len(distm) > 1 and np.min(distm) < np.inf:
            # extracting coordinates of clusters with the minimum distance
            min_coord = np.where(distm == np.min(distm))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]

            # checking the size of clusters to see if they should be merged
            if (len(clusters[merge_a]) < self.pop
                    or len(clusters[merge_b]) < self.pop):
                # if both clusters are small, do the merge
                clusters[merge_a] = np.hstack([clusters[merge_a],
                                               clusters[merge_b]])
                del clusters[merge_b]
                # update the distance matrix accordingly
                distm[merge_a] = np.min(np.vstack([distm[merge_a], distm[merge_b]]),
                                     axis=0)
                distm[:, merge_a] = distm[merge_a]
                # remove columns
                distm = np.delete(distm, merge_b, axis=0)
                distm = np.delete(distm, merge_b, axis=1)
                # fix the diagonal entries
                for idx in range(len(distm)):
                    distm[idx, idx] = np.inf
            else:
                # otherwise find principal directions
                with warnings.catch_warnings():
                    pca_a = PCA(n_components=1).fit(train_border[clusters[merge_a]])
                    pca_b = PCA(n_components=1).fit(train_border[clusters[merge_b]])
                # extract the angle of principal directions
                numerator = np.dot(pca_a.components_[0], pca_b.components_[0])
                denominator = np.linalg.norm(pca_a.components_[0])
                denominator *= np.linalg.norm(pca_b.components_[0])
                angle = abs(numerator/denominator)
                # check if angle if angle is above a specific threshold
                if angle > self.thres:
                    # do the merge
                    clusters[merge_a] = np.hstack([clusters[merge_a],
                                                   clusters[merge_b]])
                    del clusters[merge_b]
                    # update the distance matrix acoordingly
                    distm[merge_a] = np.min(np.vstack([distm[merge_a], distm[merge_b]]),
                                         axis=0)
                    distm[:, merge_a] = distm[merge_a]
                    # remove columns
                    distm = np.delete(distm, merge_b, axis=0)
                    distm = np.delete(distm, merge_b, axis=1)
                    # fixing the digaonal entries
                    for idx in range(len(distm)):
                        distm[idx, idx] = np.inf
                else:
                    # changing the distance of clusters to fininte
                    distm[merge_a, merge_b] = np.inf
                    distm[merge_b, merge_a] = np.inf

        return clusters


    def fit_sample(self, train, train_labels):
        
        classes, count_class = np.unique(train_labels, return_counts = True)
        self.class_min = classes[0] if count_class[0] < count_class[1] else classes[1]
        class_maj = classes[0] if count_class[0] > count_class[1] else classes[1]
        ind_min = np.where(train_labels == self.class_min)[0]
        train_min = train[train_labels == self.class_min]
        train_maj = train[train_labels == class_maj]
        nb_data_to_add = int((len(train_maj) - len(train_min))*self.proportion)
   
        train_border, train_non_border = self.determine_border_non_border(train, train_labels, train_min)

        if len(train_border) == 0:
            return self.return_copies(train, labels, "X_border is empty")

        clusters = self.do_the_clustering(train_border)

        vectors = [train_border[c] for c in clusters if len(c) > 0]
        if len(train_non_border) > 0:
            vectors.append(train_non_border)

        cluster_sizes = np.array([len(vect) for vect in vectors])
        cluster_density = cluster_sizes/np.sum(cluster_sizes)

        samples = []
        for ind, clust in enumerate(vectors):
            if len(clust) > 2:
                nn = NearestNeighbors(n_neighbors = np.min([self.n_neighbors + 1, len(clust)-1]))
                nn.fit(clust)
                kneighbors = nn.kneighbors()[1]

                ind_select = np.array(random.choices(range(len(kneighbors)), k=max(1, int(nb_data_to_add*cluster_density[ind]))))
                select = kneighbors[ind_select]

                steps = np.array([[random.uniform(0, 1) for _ in range(train.shape[1])] for e in range(len(select[:,0]))])

                samples.append(clust[np.array(select[:,0])] + steps * (clust[np.array(select[:,1])] - clust[np.array(select[:,0])]))
                
        if samples != []:
            samples = np.concatenate(samples, axis=0 )
            return np.concatenate((train, samples)), np.concatenate((train_labels, [self.class_min]*len(samples)))
        
        return train, train_labels