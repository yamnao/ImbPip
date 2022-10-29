from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import itertools
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import higra as hg

class dbsmote():
    '''
    Reference:
    https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_dbsmote.py
    '''
    
    def __init__(self, nb_add=1, eps=0.8, min_samples=3, n_jobs=None):
        self.nb_add = nb_add
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        
    def dijkstra(self, graph, edge_weights, centroid):
        dist, prev = np.array([np.inf]*graph.num_vertices()), np.array([np.inf]*graph.num_vertices())
        vertex = [v for v in graph.vertices()]
        Q = [v for v in graph.vertices()]
        dist[centroid] = 0
        
        while len(Q) != 0:
            ind_s = np.array(dist[Q]).argmin(axis=0)
            u = Q[ind_s]
            Q.pop(ind_s)
            
            for e in graph.in_edges(u):
                if e[0] in Q:
                    alt = dist[u] + edge_weights[e[2]]
                    if alt < dist[e[0]] and dist[u] != np.inf:
                        dist[e[0]] = alt
                        prev[e[0]] = u
        return dist, prev
    
    def find_shortest_path(self, dist, prev, pts, centroid):
        S = []
        if prev[pts] != np.inf or pts == centroid:
            while pts != np.inf:
                S.append(pts)
                pts = prev[int(pts)]
        return np.array(S, dtype=int)
    
    def get_parameters_combination(self):
        dico = {'nb_add': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                'eps':[0.2, 0.5, 0.8, 1.2],
                'min_samples':[1,3,5]
               }
        keys, values = zip(*dico.items())
        return np.array([dict(zip(keys, v)) for v in itertools.product(*values)])
    
    def fit_sample(self, data, labels):
        classes, count_class = np.unique(labels, return_counts = True)
        class_min = classes[0] if count_class[0] < count_class[1] else classes[1]
        class_maj = classes[0] if count_class[0] > count_class[1] else classes[1]
        
        nb_data_to_add = int((count_class[class_maj] - count_class[class_min])*self.nb_add)
        
        data_ = StandardScaler().fit(data).transform(data)
        data_min = data_[labels == class_min]
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=self.n_jobs).fit(data_min)
        labels_clust = db.labels_
        samples = []
        
        for label_clust in np.unique(labels_clust):
            ind_data = np.array(np.where(labels_clust == label_clust)[0])
            data_clust = data_min[ind_data]
            centroid = NearestNeighbors(n_neighbors=len(data_clust), metric='euclidean', n_jobs=1).fit(data_clust).kneighbors(np.mean(data_clust, axis=0).reshape(1, -1))[1][0][0]
            
            if len(data_clust) > 2:
                params = min(len(data_clust)-1,5)
                graph, edge_weights = hg.make_graph_from_points(data_clust, graph_type='knn+mst', n_neighbors=params)

                dist, prev = self.dijkstra(graph, edge_weights, centroid)

                for pts in range(len(ind_data)):
                    if pts != centroid:    
                        sequence = self.find_shortest_path(dist, prev, pts, centroid)
                        if int(self.nb_add) != 0 and len(sequence) != 0:
                            select_aleatoire = random.choices(range(len(sequence)), k=int(self.nb_add))
                            if select_aleatoire != []:
                                source_select = sequence[np.array(select_aleatoire)]
                                target_select = sequence[np.array(select_aleatoire)-1]

                                steps = np.array([[random.uniform(0, 1) for _ in range(data.shape[1])] for e in range(len(source_select))])
                                samples.append(data_clust[source_select] + steps * (data_clust[target_select] - data_clust[source_select]))
        samples = np.array(samples).reshape(-1, data.shape[1])
        
        return np.vstack([data_, samples]), np.hstack([labels, np.repeat(class_min, len(samples))])
        
    
        
    
        
        