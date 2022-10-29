import numpy as np
import higra as hg
import random
import itertools

class wssmote():
    def __init__(self, nb_add=1, nb_neigh=5, interpolation='star'):
        self.nb_add = nb_add
        self.interpolation = interpolation
        self.nb_neigh = nb_neigh
        
    def get_parameters_combination(self):
        dico = {'nb_add': [0.2, 0.5, 0.75, 1.0, 1.5, 2.0], 
                'interpolation': ['star', 'mesh', 'inside_5', 'inside_10', 'inside_20'], 
               'nb_neigh':[2, 5, 7, 10]}
        keys, values = zip(*dico.items())
        return np.array([dict(zip(keys, v)) for v in itertools.product(*values)])
    
    def watershed_clustering(self):
        labels_clust = np.array(hg.labelisation_watershed(self.graph, self.edge_weights))
        return labels_clust
    
    def watershed_clustering_complex(self, nb_clust):
        label_clust = np.array(hg.labelisation_watershed(self.graph, self.edge_weights))
        rag = hg.make_region_adjacency_graph_from_labelisation(self.graph, label_clust)
        index = np.where(rag.edge_map != -1)[0]
        source, target, poids = self.graph.edge_list()[0][index], self.graph.edge_list()[1][index], self.edge_weights[index]
        dict_nb_rag = {}
        for node in np.unique(np.concatenate((source, target))):
            dict_nb_rag[node] = [label_clust[node], 0]
        index_edge_tri = np.argsort(poids)
        ind = 0 
        while ind < (len(index_edge_tri)-1):
            if dict_nb_rag[source[index_edge_tri[ind]]][0] != dict_nb_rag[target[index_edge_tri[ind]]][0]:
                if dict_nb_rag[source[index_edge_tri[ind]]][1] <  nb_clust and dict_nb_rag[target[index_edge_tri[ind]]][1] < nb_clust:
                    value1 = dict_nb_rag[source[index_edge_tri[ind]]][0]
                    value2 = dict_nb_rag[target[index_edge_tri[ind]]][0]
                    for k in dict_nb_rag.keys():
                        if dict_nb_rag[k][0] == value1 or dict_nb_rag[k][0] == value2:
                            dict_nb_rag[k][1] = dict_nb_rag[k][1] + 1
                            dict_nb_rag[k][0] = value2
                    label_clust[label_clust == value1] = value2
            ind +=1  
        return label_clust
 
    def fit_sample(self, data, labels):
        classes, count_class = np.unique(labels, return_counts = True)
        self.class_min = classes[0] if count_class[0] < count_class[1] else classes[1]
        self.class_maj = classes[0] if count_class[0] > count_class[1] else classes[1]
        
        self.nb_data_to_add = int((count_class[self.class_maj] - count_class[self.class_min])*self.nb_add)
        
        ind_min = np.where(labels == self.class_min)[0]
        self.data_min = data[labels == self.class_min]
        self.data_maj = data[labels == self.class_maj]
        
        self.graph, self.edge_weights = hg.make_graph_from_points(self.data_min, graph_type='knn+mst', n_neighbors=self.nb_neigh)
        
         
        labels_clust = self.watershed_clustering()
       
        rag = hg.make_region_adjacency_graph_from_labelisation(self.graph, labels_clust)
        index_real = np.array(np.where(rag.edge_map!=-1)[0])
        source, target = self.graph.edge_list()[0][index_real], self.graph.edge_list()[1][index_real]
        
        data_conv = self.data_min[np.unique(np.concatenate((source, target)))]

        samples = []
        if self.interpolation == 'star':
            X_mean = np.mean(data_conv, axis=0)
            k = max([1, int(np.rint(self.nb_data_to_add/len(data_conv)))])
            samples = []
            for x in data:
                diff = X_mean - x
                for i in range(1, k+1):
                    samples.append(x + float(i)/(k+1)*diff)
            return np.concatenate((data, samples)), np.concatenate((labels, [self.class_min]*len(samples)))
        
        elif self.interpolation == 'mesh':
            k = max([1, int(np.rint(self.nb_data_to_add/len(data_conv)))])
            for i in range(len(source)):
                diff = self.data_min[source[i]] - self.data_min[target[i]]
                for li in range(1, k+1):
                    samples.append(self.data_min[target[i]] + float(li)/(k+1)*diff)
            return np.concatenate((data, samples)), np.concatenate((labels, [self.class_min]*len(samples)))
        
        elif self.interpolation == 'inside_5':
            labels_clust = self.watershed_clustering_complex(nb_clust=5)
            k = max([1, int(np.rint(self.nb_data_to_add/len(self.data_min)))])
            for l_c in np.unique(labels_clust):
                data_c = self.data_min[labels_clust == l_c]
                for x in data_c:
                    random_j = random.choices(range(len(data_c)), k=1)
                    diff = data_c[random_j] - x
                    for li in range(1, k+1):
                        samples.append((x + float(li)/(k+1)*diff)[0])
            return np.concatenate((data, samples)), np.concatenate((labels, [self.class_min]*len(samples)))
                
            
        elif self.interpolation == 'inside_10':
            labels_clust = self.watershed_clustering_complex(nb_clust=10)
            k = max([1, int(np.rint(self.nb_data_to_add/len(self.data_min)))])
            for l_c in np.unique(labels_clust):
                data_c = self.data_min[labels_clust == l_c]
                for x in data_c:
                    random_j = random.choices(range(len(data_c)), k=1)
                    diff = data_c[random_j] - x
                    for li in range(1, k+1):
                        samples.append((x + float(li)/(k+1)*diff)[0])
            return np.concatenate((data, samples)), np.concatenate((labels, [self.class_min]*len(samples)))
            
        elif self.interpolation == 'inside_20':
            labels_clust = self.watershed_clustering_complex(nb_clust=20)
            k = max([1, int(np.rint(self.nb_data_to_add/len(self.data_min)))])
            for l_c in np.unique(labels_clust):
                data_c = self.data_min[labels_clust == l_c]
                for x in data_c:
                    random_j = random.choices(range(len(data_c)), k=1)
                    diff = data_c[random_j] - x
                    for li in range(1, k+1):
                        samples.append((x + float(li)/(k+1)*diff)[0])
            return np.concatenate((data, samples)), np.concatenate((labels, [self.class_min]*len(samples)))
        