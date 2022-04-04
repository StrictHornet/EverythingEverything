# IMPORT LIBRARIES UTILISED IN THIS SOLUTION
#import warnings
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------
#K-CLUSTERING OBJECT INSTANTIATION CLASS
#--------------------------------------------
class k_clustering_class():

    # INITIALIZATION OF OBJECT INPUT DATASET, K NUMBER OF CLUSTERS AND NUMBER OF ITERATIONS
    def __init__(self, dataset, k, iteration_count):
        self.k = k
        self.clusters = []
        self.iteration_count = iteration_count # Iteration count to mitigate infinite loop
        self.centroids = self._centroid_rep(dataset)
    # RANDOMLY CHOOSING CLUSTER CENTROID REPRESENTATIVES
    def _centroid_rep(self, dataset):  
        c_rep = []
        i = np.random.choice(len(dataset), self.k, replace=False)
        for j in i:
           c_rep.append(dataset[j])
        return c_rep
   
    # THIS FUNCTION INITIALISES CENTROIDS
    def _init_centroids(self, dataset, mean_clustering=True):
        self.clusters = []
        for object in dataset:
            centroid_dot = self.centroids[0]
            c_dist = np.Inf
            # CALCULATE EUCLIDEAN OR MANHATTAN DISTANCES BASED ON K TYPE OF CLUSTERING
            for centroid in self.centroids:
                new_dist = _euclid_dist(object, centroid) if mean_clustering else _manhattan_dist(object, centroid)

                if new_dist < c_dist:
                    c_dist = new_dist
                    centroid_dot = centroid

            # APPEND CENTROIDS
            self.clusters.append(centroid_dot)

    # UPDATE CENTROIDS TOWARDS IDEAL POSITION
    def _update_centroids(self, dataset, mean_clustering =True):
        temp_storage = []
        k_clusters = len(self.clusters)

        # STORE EACH CENTROID VALUES
        for inst in self.centroids:
            # INITIALISE ARRAY TO STORE DATASET CONTENT
            dataset_content = []
            for k in range(k_clusters):
                if (self.clusters[k] == inst).all():
                    dataset_content.append(dataset[k])
            temp_storage.append(dataset_content)
        update_centroids = []
        for item in temp_storage:
            # CALCULATE MEAN OR MEDIAN DEPENDING ON mean_clustering value. *if mean_clustering is false; median*
            centroid_update_values = np.mean(item, axis=0) if mean_clustering else np.median(item, axis=0)
            update_centroids.append(centroid_update_values)
        self.centroids = update_centroids

    # DISTINGUISH AND ADD LABELS TO DATASET           
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
                    embedding_lb = dataset[i][:-1]
                    label = dataset[i][-1]
                    total_clusters_arr.append(embedding_lb)
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

    # F-SCORE  CALCULATOR
    def _f_score(self, class_precision, class_recall):
        f_score = 0
        if (class_precision + class_recall) != 0:
            f_score = (2 * class_precision * class_recall) / (class_precision + class_recall)
        return f_score

    # CALCULATE THE PRECISION, RECALL AND F-SCORE FOR EACH CLUSTER CLASSIFICATION
    def _B_CUBED_(self, dataset, total_clusters, no_of_animals, no_of_countries, no_of_fruits, no_of_veggies):
        precision, recall, f_score = 0, 0, 0
        k_clusters = len(total_clusters)
        for i in range(k_clusters):
            # CALCULATE PRECISION
            if len(total_clusters[i]) != 0:
                animal_precision = no_of_animals[i] / len(total_clusters[i])
                country_precision = no_of_countries[i] / len(total_clusters[i])
                fruit_precision= no_of_fruits[i] / len(total_clusters[i])
                veggie_precision = no_of_veggies[i] / len(total_clusters[i])
            
            #CALCULATE RECALL
            if len(total_clusters[i]) != 0:
                animal_recall = no_of_animals[i] / sum(no_of_animals)
                country_recall = no_of_countries[i] / sum(no_of_countries)
                fruit_recall = no_of_fruits[i] / sum(no_of_fruits)
                veggie_recall = no_of_veggies[i] / sum(no_of_veggies)
            
            # CALL F-SCORE CALCULATOR
            animal_f_score = self._f_score(animal_precision, animal_recall)
            country_f_score = self._f_score(country_precision, country_recall)
            fruit_f_score = self._f_score(fruit_precision, fruit_recall)
            veggie_f_score = self._f_score(veggie_precision, veggie_recall)

            precision += ((animal_precision * no_of_animals[i]) + (country_precision * no_of_countries[i]) + (fruit_precision* no_of_fruits[i]) + (veggie_precision * no_of_veggies[i])) / len(dataset)
            recall += ((animal_recall * no_of_animals[i]) + (country_recall * no_of_countries[i]) + (fruit_recall* no_of_fruits[i]) + (veggie_recall * no_of_veggies[i])) / len(dataset)
            f_score += ((animal_f_score * no_of_animals[i]) + (country_f_score * no_of_countries[i]) + (fruit_f_score* no_of_fruits[i]) + (veggie_f_score * no_of_veggies[i])) / len(dataset)

        return precision, recall, f_score

#--------------------------------------------
# ADDITIONAL FUNCTIONS USED FOR IMPLEMETATION
#--------------------------------------------
np.random.seed(22)
maximum_k_clusters = 9

# CALCULATE EUCLIDEAN DISTANCE FOR K-MEANS
def _euclid_dist(v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

# CALCULATE MANHATTAN DISTANCE FOR K-MEDIANS
def _manhattan_dist(v1, v2):
        return np.sum(np.abs(v1 - v2))

# L2 NORMALIZATION OF DATASET
def _l2_norm(dataset):
    normalized_dataset = []
    for data in dataset:
        label = data[-1]
        data = data[:-1]
        sub_dataset = [] 
        for value in data:
            sub_dataset.append(value/np.linalg.norm(data))
        sub_dataset.append(label)
        normalized_dataset.append( sub_dataset)
    return np.array(normalized_dataset) 

# GET CLUSTERING DATA
def _get_datasets(bool_l2_norm = False):
    animal_cluster_values = np.genfromtxt("animals", delimiter=' ')[:, 1:]
    animal_cluster_label = np.array([[1]] * len(animal_cluster_values))
    animal_cluster_values = np.append(animal_cluster_values, animal_cluster_label, axis=1)

    country_cluster_values = np.genfromtxt("countries", delimiter=' ')[:, 1:]
    country_cluster_label = np.array([[2]] * len(country_cluster_values))
    country_cluster_values = np.append(country_cluster_values, country_cluster_label, axis=1)

    fruit_cluster_values = np.genfromtxt("fruits", delimiter=' ')[:, 1:]
    fruit_cluster_label = np.array([[3]] * len(fruit_cluster_values))
    fruit_cluster_values = np.append(fruit_cluster_values, fruit_cluster_label, axis=1)

    veggie_cluster_values = np.genfromtxt("veggies", delimiter=' ')[:, 1:]
    veggie_cluster_label = np.array([[4]] * len(veggie_cluster_values))
    veggie_cluster_values = np.append(veggie_cluster_values, veggie_cluster_label, axis=1)

    dataset = np.concatenate([animal_cluster_values, country_cluster_values, fruit_cluster_values, veggie_cluster_values], axis=0).astype(float)
    l2_dataset_holder = dataset
    # NORMALIZE DATASET *if bool_l2_norm is True*
    if bool_l2_norm:
        return _l2_norm(l2_dataset_holder)
    else:
        return dataset 

# PLOTS B-CUBED PRECISION, RECALL, F-SCORE VALUES
def _plot_graphs(array_precision, array_recall, array_f_score, mean_clustering = True, bool_l2_norm = False):
    abscissa = np.arange(1, len(array_precision) + 1, 1)
    precision_ordinates = array_precision
    recall_ordinates = array_recall
    f_score_ordinates = array_f_score
    
    if mean_clustering:
        if not bool_l2_norm:
            plt.title('k-Means Clustering (No L2 Normalization)')
        else:
            plt.title('L2 Normalized k-Means Clustering')
    else:
        if not bool_l2_norm:
            plt.title('k-Medians Clustering (No L2 Normalization)')
        else:
            plt.title('L2 Normalized k-Medians Clustering')
    
    # PROPERTIES OF PLOTTED GRAPHS
    plt.xlabel('k-clusters')
    plt.ylabel('B-CUBED Values [precision, recall, f-score]')
    plt.plot(abscissa, f_score_ordinates, color='yellow', marker='$...$', label='F-score')
    plt.plot(abscissa, recall_ordinates, color='green', marker='+', label='Recall')
    plt.plot(abscissa, precision_ordinates, color='red', marker='2',label='Precision')
    plt.legend(loc='lower right')
    plt.show()

# K-CLUSTERING FUNCTION TO RUN CLUSTERING MODEL
def _k_cluster(mean_clustering = True, bool_l2_norm = False):
    data = _get_datasets(bool_l2_norm)
    array_precision = []
    array_f_score = []  
    array_recall = []

    print('K-Clusters', 'B-CUBED Precision', 'B-CUBED Recall', 'B-CUBED F-score')

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
            precision, recall, f_score = k_clustering_object._B_CUBED_(data, total_clusters, no_of_animals, no_of_countries, no_of_fruits, no_of_veggies)  
            count_variable += 1

        array_f_score.append(f_score)
        array_precision.append(precision)
        array_recall.append(recall)
        
        print(k, "         ||" , f'{precision:.4f}', "     ||" , f'{recall:.4f}', "    ||" , f'{f_score:.4f}')
        k += 1
        
    # PLOT RESULTS
    _plot_graphs(array_precision, array_recall, array_f_score, mean_clustering, bool_l2_norm)
    print()

# RUNS MEAN & MEADIAN CLUSTERING WITH AND WITHOUT L2 NORMALIZATION
def _run_k_means_and_median_clustering():
    print("SOLUTION:\n")

    print('Q3: Mean B-CUBED Clustering [No Normalization]\n')
    _k_cluster(bool_l2_norm=False)

    print('Q4: Mean B-CUBED Clustering Normalized\n')
    _k_cluster(bool_l2_norm=True)

    print('Q5: Median B-CUBED Clustering [No Normalization]\n')
    _k_cluster(mean_clustering=False, bool_l2_norm=False)

    print('Q6: Median B-CUBED Clustering Normalized\n')
    _k_cluster(mean_clustering=False, bool_l2_norm=True)

# RUN CLUSTERING ON CLASSES
_run_k_means_and_median_clustering()
### THANK YOU FOR TEACHING! ###