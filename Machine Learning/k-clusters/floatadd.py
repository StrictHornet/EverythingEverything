import numpy as np
import matplotlib.pyplot as plt

SEED = 101
np.random.seed(SEED)
maxCluster = 9
def loadData(file_name, label):
    data = np.loadtxt(file_name, dtype=str, delimiter=' ')
    data = data[:, 1:]
    # label dataset
    y = np.array([[label]] * len(data))
    data = np.append(data, y, axis=1)
    return data

def loadDataset(l2_Norm = False):
    c1 = loadData("animals", 1)
    c2 = loadData("countries", 2)
    c3 = loadData("fruits", 3)
    c4 = loadData("veggies", 4)
    dataset = np.concatenate([c1, c2, c3, c4], axis=0).astype(float)
    if l2_Norm:
        return l2_Normalization(dataset)
    else:
        return dataset


def l2_Normalization(dataset):
    normalizedData = []

    for item in dataset:
        itemDataset = []
        label, item = item[-1], item[:-1] 
        for attribute in item:
            itemDataset.append(attribute/np.linalg.norm(item))
        itemDataset.append(label)
        normalizedData.append(itemDataset)
    return np.array(normalizedData) 

def runModel(KMeans = True, l2_Norm = False):
    data = loadDataset(l2_Norm)
    precisionValues = []
    fscoreValues = []  
    recallValues = []

    
    print('k', 'Precision', 'Recall', 'F-Score')

    for k in range(maxCluster):
        cluster = KClusterModel(data, k+1, 100)
        count  = 0
        fscore, precision, recall = None, None, None

        while count < cluster.maxIter:
            cluster.selectCentroid(data, KMeans)
            allClusters, animalCount, countryCount, fruitCount, vegCount = cluster.addLabel(data)
            cluster.optimizeCentroid(data, KMeans)
            precision, recall, fscore = cluster.bCubed(data, allClusters, animalCount, countryCount, fruitCount, vegCount)##############################    
            count+=1

        fscoreValues.append(fscore)
        precisionValues.append(precision)
        recallValues.append(recall)

        # print(precisionValues)
        # print evaluation metrics
        
        print(k + 1, round(precision, 2), round(recall, 2), round(fscore, 2))
        
        # plot evaluation metrics 
        plots(precisionValues, recallValues, fscoreValues, KMeans, l2_Norm)
        print()

def plots(precisionValues, recallValues, fscoreValues, KMeans = True, l2_Norm = False):
    x = np.arange(1, len(precisionValues) + 1, 1)
    y_1, y_2, y_3 = precisionValues, recallValues, fscoreValues
    if KMeans:
        if l2_Norm:
            plt.title('k-Means with L2 Normalization')
        else:
            plt.title('k-Means without L2 Normalization')
    else:
        if l2_Norm:
            plt.title('k-Medians with L2 Normalization')
        else:
            plt.title('k-Medians without L2 Normalization')
    
    plt.xlabel('k')
    plt.ylabel('B-CUBED')
    plt.plot(x, y_1, color='tab:blue', marker='o',label='Precision')
    plt.plot(x, y_2, color='tab:orange', marker='D', label='Recall')
    plt.plot(x, y_3, color='tab:green', marker='^', label='FScore')
    plt.legend(loc='lower right')
    plt.show()

def getEuclideanDistance(X, Y):
        return np.sqrt(np.sum(np.square(X - Y)))

def getManhattanDistance(X, Y):
        return np.sum(np.abs(X - Y))


class KClusterModel():
    def __init__(self, data, k, maxIter):
        self.k = k
        self.maxIter = maxIter
        self.clusters = []
        self.centroids = self.getGroupReps(data)
 
    def getGroupReps(self, data):  
        representatives = []
        i = np.random.choice(len(data), self.k, replace=False)
        for j in i:
           representatives.append(data[j])
        return representatives
   
    def selectCentroid(self, data, kMeans=True):
        self.clusters = []
        for X in data:
            point = self.centroids[0]
            # find the closest mean
            distanceToClosestRep = np.Inf
            for centroid in self.centroids:
                if kMeans:
                   
                    distance = getEuclideanDistance(X, centroid)#############
                else:
                    
                    distance = getManhattanDistance(X, centroid)
                if distance < distanceToClosestRep:
                    distanceToClosestRep = distance
                    point = centroid
            # assign X to its closest representative, each X will have its new centroid
            self.clusters.append(point)

    def optimizeCentroid(self, data, KMeans =True):
        everyGroup = []
        noOfClusters = len(self.clusters)

        # loop through array storing values of Y
        for Y in self.centroids:
            # initialize group array
            group = []
            for k in range(noOfClusters):
                if (self.clusters[k] == Y).all():
                    # append element of dataset to group array
                    group.append(data[k])
            # append group to everyGroup
            everyGroup.append(group)
        updatedCentroids = []
        for item in everyGroup:
            # calculate mean for k-means ; median for k-median
            updatedCentroid = np.mean(item, axis=0) if KMeans else np.median(item, axis=0)
            updatedCentroids.append(updatedCentroid)
        self.centroids = updatedCentroids
               
    def addLabel(self, data):
        noOfClusters = len(self.clusters)
        allClusters, animalCount, countryCount, fruitCount, vegCount  = [],[], [], [], []

        for centroid in self.centroids:
            group = []
            noOfAnimals, noOfCountries, noOfFruits, noOfVeggies = 0, 0, 0, 0

            for i in range(noOfClusters):
                if (self.clusters[i]==centroid).all():
                    x = data[i][:-1]
                    y = data[i][-1]
                    group.append(x)
                    if y == 1:
                        noOfAnimals += 1
                    elif y == 2:
                        noOfCountries += 1
                    elif y == 3:
                        noOfFruits += 1
                    else:
                        noOfVeggies += 1
            animalCount.append(noOfAnimals)
            countryCount.append(noOfCountries)
            fruitCount.append(noOfFruits)
            vegCount.append(noOfVeggies)
            allClusters.append(group)
        return allClusters, animalCount, countryCount, fruitCount, vegCount

    def bCubed(self, data, allClusters, animalCount, countryCount, fruitCount, vegCount):
        precision, recall, fscore = 0, 0, 0
        noOfClusters = len(allClusters)
        for i in range(noOfClusters):
            animalPrecision = countryPrecision = fruitPrecision = vegPrecision = 0
            animalRecall = countryRecall = fruitRecall = vegRecall = 0
            animalFscore, countryFscore, fruitFscore, vegFscore = 0, 0, 0, 0

            if len(allClusters[i]) != 0:
                animalPrecision = animalCount[i] / len(allClusters[i])
                countryPrecision = countryCount[i] / len(allClusters[i])
                fruitPrecision= fruitCount[i] / len(allClusters[i])
                vegPrecision = vegCount[i] / len(allClusters[i])

                animalRecall = animalCount[i] / sum(animalCount)
                countryRecall = countryCount[i] / sum(countryCount)
                fruitRecall = fruitCount[i] / sum(fruitCount)
                vegRecall = vegCount[i] / sum(vegCount)
           
            if (animalPrecision + animalRecall) != 0:
                animalFscore = 2 * animalPrecision * animalRecall / (animalPrecision + animalRecall)
           
            if (countryPrecision + countryRecall) != 0:
                countryFscore = 2 * countryPrecision * countryRecall / (countryPrecision + countryRecall)
         
            if (fruitPrecision + fruitRecall) != 0:
                fruitFscore = 2 * fruitPrecision* fruitRecall / (fruitPrecision+ fruitRecall)
         
            if (vegPrecision + vegRecall) != 0:
                vegFscore = 2 * vegPrecision * vegRecall / (vegPrecision + vegRecall)

            precision += (animalPrecision * animalCount[i]) / len(data) + \
                         (countryPrecision * countryCount[i]) / len(data) + \
                         (fruitPrecision* fruitCount[i]) / len(data) + \
                         (vegPrecision * vegCount[i]) / len(data)

            recall += (animalRecall * animalCount[i]) / len(data) + \
                      (countryRecall * countryCount[i]) / len(data) + \
                      (fruitRecall * fruitCount[i]) / len(data) + \
                      (vegRecall * vegCount[i]) / len(data)

            fscore += (animalFscore * animalCount[i]) / len(data) + \
                      (countryFscore * countryCount[i]) / len(data) + \
                      (fruitFscore * fruitCount[i]) / len(data) + \
                      (vegFscore * vegCount[i]) / len(data)
        return precision, recall, fscore
        




print('\nQuestion 3: k-means B-CUBED Clustering without normalization')
runModel(l2_Norm=False)
print('\nQuestion 4: k-means B-CUBED Clustering with normalization')
runModel(l2_Norm=True)
print('\nQuestion 5: k-medians B-CUBED Clusteringwithout normalization')
runModel(KMeans=False, l2_Norm=False)
print('\nQuestion 6: k-medians B-CUBED Clustering with normalization')
runModel(KMeans=False, l2_Norm=True)