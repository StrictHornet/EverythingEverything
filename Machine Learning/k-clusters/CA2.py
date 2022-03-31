# Import libraries used in solution
import numpy as np
import pandas as pd
#import warnings
import matplotlib.pyplot as plt
'''
# Import data to numpy array, ignore first value from array
animal_file = np.genfromtxt('animals', delimiter=" ")[:, 1:]
countries_file = np.genfromtxt('countries', delimiter=" ")[:,1:]
fruits_file = np.genfromtxt('fruits', delimiter=" ")[:,1:]
veggies_file = np.genfromtxt('veggies', delimiter=" ")[:,1:]

'''
'''


'''

#K-CLUSTERING OBJECT CLASS
class k_clustering_class():

    #INITIALIZATION OF OBJECT INPUT DATASET, K NUMBER OF CLUSTERS AND NUMBER OF ITERATIONS
    def __init__(self, dataset, k, iteration_count):
        self.k = k
        self.clusters = []
        self.iteration_count = iteration_count # Iteration count to mitigate infinite loop
        self.centroids = self._centroid_rep(dataset)
    #-----------------------------------------------------------------------------------------------------------------add comment
    def _centroid_rep(self, dataset):  
        c_rep = []
        i = np.random.choice(len(dataset), self.k, replace=False)
        for j in i:
           c_rep.append(dataset[j])
        return c_rep
   
   # THIS FUNCTION SELECTS INITIAL CENTROIDS
    def _init_centroids(self, dataset, mean_clustering=True):
        self.clusters = []
        for X in dataset:
            centroid_dot = self.centroids[0]
            # FIND CLOSEST MEAN DISTANCE
            c_rep_dist = np.Inf
            # CALCULATE EUCLIDEAN OR MANHATTAN DISTANCES BASED ON K TYPE OF CLUSTERING
            for centroid in self.centroids:
                new_dist = euclid_dist(X, centroid) if mean_clustering else manhattan_dist(X, centroid)

                if new_dist < c_rep_dist:
                    c_rep_dist = new_dist
                    centroid_dot = centroid

            # ASSIGN X TO CENTROIDS FOR CLUSTERINGS
            self.clusters.append(centroid_dot)

    # UPDATE CENTROIDS TOWARDS IDEAL POSITION
    def _update_centroids(self, dataset, mean_clustering =True):
        temp_storage = []
        k_clusters = len(self.clusters)

        # STORE EACH CENTROID VALUES
        for _ in self.centroids:
            # INITIALISE ARRAY TO STORE DATASET CONTENT
            dataset_content = []
            for k in range(k_clusters):
                if (self.clusters[k] == _).all():
                    dataset_content.append(dataset[k])
            temp_storage.append(dataset_content)
        update_centroids = []
        for item in temp_storage:
            # CALCULATE MEAN OR MEDIAN DEPENDING ON mean_clustering value. *if mean_clustering is false; median*
            centroid_update_values = np.mean(item, axis=0) if mean_clustering else np.median(item, axis=0)
            update_centroids.append(centroid_update_values)
        self.centroids = update_centroids

    #--------------------------------------------------------------------------------------------------------------add comment           
    def _labels_append(self, dataset):
        k_clusters = len(self.clusters)
        total_clusters = []
        no_of_animals = []
        no_of_countries = []
        no_of_fruits = []
        no_of_veggies = []

        for centroid in self.centroids:
            total_clusters_arr = []
            animal_total = 0
            country_total = 0
            fruit_total = 0
            veggie_total = 0

            for i in range(k_clusters):
                if (self.clusters[i]==centroid).all():
                    x_embeddings = dataset[i][:-1]
                    label = dataset[i][-1]
                    total_clusters_arr.append(x_embeddings)
                    if label == 1:
                        animal_total += 1
                    elif label == 2:
                        country_total += 1
                    elif label == 3:
                        fruit_total += 1
                    else:
                        veggie_total += 1
            no_of_animals.append(animal_total)
            no_of_countries.append(country_total)
            no_of_fruits.append(fruit_total)
            no_of_veggies.append(veggie_total)
            total_clusters.append(total_clusters_arr)
        return total_clusters, no_of_animals, no_of_countries, no_of_fruits, no_of_veggies

    # CALCULATE THE PRECISION, RECALL AND F-SCORE FOR EACH CLUSTER CLASSIFICATION
    def _B_CUBED_(self, dataset, total_clusters, no_of_animals, no_of_countries, no_of_fruits, no_of_veggies):
        precision, recall, f_score = 0, 0, 0
        k_clusters = len(total_clusters)
        for i in range(k_clusters):
            #animal_precision, country_precision, fruit_precision, veggie_precision = 0, 0, 0, 0
            #animal_recall , ountry_recall, fruit_recall, veggie_recall = 0, 0, 0, 0
            animal_f_score, country_f_score, fruit_f_score, veggie_f_score = 0, 0, 0, 0

            if len(total_clusters[i]) != 0:
                animal_precision = no_of_animals[i] / len(total_clusters[i])
                country_precision = no_of_countries[i] / len(total_clusters[i])
                fruit_precision= no_of_fruits[i] / len(total_clusters[i])
                veggie_precision = no_of_veggies[i] / len(total_clusters[i])

                animal_recall = no_of_animals[i] / sum(no_of_animals)
                country_recall = no_of_countries[i] / sum(no_of_countries)
                fruit_recall = no_of_fruits[i] / sum(no_of_fruits)
                veggie_recall = no_of_veggies[i] / sum(no_of_veggies)
           
            if (animal_precision + animal_recall) != 0:
                animal_f_score = 2 * animal_precision * animal_recall / (animal_precision + animal_recall)
           
            if (country_precision + country_recall) != 0:
                country_f_score = 2 * country_precision * country_recall / (country_precision + country_recall)
         
            if (fruit_precision + fruit_recall) != 0:
                fruit_f_score = 2 * fruit_precision* fruit_recall / (fruit_precision+ fruit_recall)
         
            if (veggie_precision + veggie_recall) != 0:
                veggie_f_score = 2 * veggie_precision * veggie_recall / (veggie_precision + veggie_recall)

            # precision += (animal_precision * no_of_animals[i]) / len(dataset) + \
            #              (country_precision * no_of_countries[i]) / len(dataset) + \
            #              (fruit_precision* no_of_fruits[i]) / len(dataset) + \
            #              (veggie_precision * no_of_veggies[i]) / len(dataset)

            precision += ((animal_precision * no_of_animals[i]) + (country_precision * no_of_countries[i]) + (fruit_precision* no_of_fruits[i]) + (veggie_precision * no_of_veggies[i])) / len(dataset)

            recall += (animal_recall * no_of_animals[i]) / len(dataset) + \
                      (country_recall * no_of_countries[i]) / len(dataset) + \
                      (fruit_recall * no_of_fruits[i]) / len(dataset) + \
                      (veggie_recall * no_of_veggies[i]) / len(dataset)

            f_score += (animal_f_score * no_of_animals[i]) / len(dataset) + \
                      (country_f_score * no_of_countries[i]) / len(dataset) + \
                      (fruit_f_score * no_of_fruits[i]) / len(dataset) + \
                      (veggie_f_score * no_of_veggies[i]) / len(dataset)
        return precision, recall, f_score

np.random.seed(101)
maximum_k_clusters = 9
def loadData(file_name, label):
    data = np.loadtxt(file_name, dtype=str, delimiter=' ')
    data = data[:, 1:]
    # label dataset
    y = np.array([[label]] * len(data))
    data = np.append(data, y, axis=1)
    return data

def _get_datasets(bool_l2_norm = False):
    c1 = loadData("animals", 1)
    c2 = loadData("countries", 2)
    c3 = loadData("fruits", 3)
    c4 = loadData("veggies", 4)
    dataset = np.concatenate([c1, c2, c3, c4], axis=0).astype(float)
    if bool_l2_norm:
        return _l2_norm(dataset)
    else:
        return dataset 


def _l2_norm(dataset):
    normalized_dataset = []

    for data in dataset:
        sub_dataset = []
        label, data = data[-1], data[:-1] 
        for value in data:
            sub_dataset.append(value/np.linalg.norm(data))
        sub_dataset.append(label)
        normalized_dataset.append(sub_dataset)
    return np.array(normalized_dataset) 

def _k_cluster(mean_clustering = True, bool_l2_norm = False):
    data = _get_datasets(bool_l2_norm)
    array_precision = []
    array_f_score = []  
    array_recall = []

    
    print('k-cluster', 'B-CUBED Precision', 'B-CUBED Recall', 'B-CUBED F-score')

    k = 1
    while k <= maximum_k_clusters:
        k_clustering_object = k_clustering_class(data, k, 100)
        count_variable  = 0
        f_score = None
        precision = None
        recall = None

        while count_variable < k_clustering_object.iteration_count:
            k_clustering_object._init_centroids(data, mean_clustering)
            total_clusters, no_of_animals, no_of_countries, no_of_fruits, no_of_veggies = k_clustering_object._labels_append(data)
            k_clustering_object._update_centroids(data, mean_clustering)
            precision, recall, f_score = k_clustering_object._B_CUBED_(data, total_clusters, no_of_animals, no_of_countries, no_of_fruits, no_of_veggies)##############################    
            count_variable += 1

        array_f_score.append(f_score)
        array_precision.append(precision)
        array_recall.append(recall)

        # print(array_precision)
        # print evaluation metrics
        
        print(k, round(precision, 4), round(recall, 4), round(f_score, 4))
        k += 1
        
        # plot evaluation metrics 
    plot_graphs(array_precision, array_recall, array_f_score, mean_clustering, bool_l2_norm)
    print()

def plot_graphs(array_precision, array_recall, array_f_score, mean_clustering = True, bool_l2_norm = False):
    abscissa = np.arange(1, len(array_precision) + 1, 1)
    precision_ordinates = array_precision
    recall_ordinates = array_recall
    f_score_ordinates = array_f_score
    
    if mean_clustering:
        if not bool_l2_norm:
            plt.title('k-Means Clustering (No L2 Normalisation)')
        else:
            plt.title('L2 Normalised k-Means Clustering')
    else:
        if not bool_l2_norm:
            plt.title('k-Medians Clustering (No L2 Normalisation)')
        else:
            plt.title('L2 Normalised k-Medians Clustering')
    
    plt.xlabel('k-clusters')
    plt.ylabel('B-CUBED Values [precision, recall, f-score]')
    plt.plot(abscissa, f_score_ordinates, color='yellow', marker='$...$', label='F-score')
    plt.plot(abscissa, recall_ordinates, color='green', marker='+', label='Recall')
    plt.plot(abscissa, precision_ordinates, color='red', marker='2',label='Precision')
    plt.legend(loc='lower right')
    plt.show()

def euclid_dist(v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

def manhattan_dist(v1, v2):
        return np.sum(np.abs(v1 - v2))
        

print('\nQuestion 3: k-means B-CUBED Clustering without normalization')
_k_cluster(bool_l2_norm=False)
print('\nQuestion 4: k-means B-CUBED Clustering with normalization')
_k_cluster(bool_l2_norm=True)
print('\nQuestion 5: k-medians B-CUBED Clustering without normalization')
_k_cluster(mean_clustering=False, bool_l2_norm=False)
print('\nQuestion 6: k-medians B-CUBED Clustering with normalization')
_k_cluster(mean_clustering=False, bool_l2_norm=True)


#Compare results from OG edition to final edition