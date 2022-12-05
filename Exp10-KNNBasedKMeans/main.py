import numpy as np
import math
from matplotlib import pyplot as plt

class KnnKMeans:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.clusters = []
        for i in range(n_clusters):
            self.clusters.append(set())

    def fit(self, X_train, threshold):
        cluster_index = 0
        for point in X_train:
            closest_cluster_index = 0
            min_distance = math.inf
            found = False
            i = 0
            while i<=cluster_index and i<self.n_clusters:
                if(len(self.clusters[i])==0):
                    i+=1
                    continue
                tmp = math.inf
                for pt in self.clusters[i]:
                    dist = self.calculate_distance(point, pt)
                    if(dist<tmp):
                        tmp = dist
                if tmp<min_distance:
                    if tmp<=threshold:
                        found = True
                    min_distance = tmp
                    closest_cluster_index = i
                i+=1
        
            if(found==True):
                self.clusters[closest_cluster_index].add(tuple(point))
            else:
                if(cluster_index<=self.n_clusters-1):
                    self.clusters[cluster_index].add(tuple(point))
                    if(cluster_index < self.n_clusters - 1):
                        cluster_index += 1
                else:
                    self.clusters[closest_cluster_index].add(tuple(point))
            print(point, "-> ", "clusters: ", self.clusters)

    def calculate_distance(self, point, data):
        # print(point, data)
        return np.sqrt(np.sum((np.array(point) - np.array(data))**2))


X_train = [[0,0], [1,7], [1,6], [1,1], [1,1], [1,2], [1,3], [1,4], [1,5]]

print("\n\n--------------------- t = 0 ---------------------")
kmeans0 = KnnKMeans(n_clusters = 3)
kmeans0.fit(X_train, 0)
X = []
Y = []
print("\nFinal clusters: ")
for cluster in kmeans0.clusters:
    print(cluster)
    temp_x = []
    temp_y = []
    for elem in cluster:
        temp_x.append(elem[0])
        temp_y.append(elem[1])
    X.append(temp_x)
    Y.append(temp_y)

plt.scatter(X[0], Y[0], color='red')
plt.scatter(X[1], Y[1], color='blue')
plt.scatter(X[2], Y[2], color='green')
plt.title("t = 0")
plt.show()

print("\n\n--------------------- t = 1 ---------------------")
kmeans1 = KnnKMeans(n_clusters = 3)
kmeans1.fit(X_train, 1)
X = []
Y = []
print("\nFinal clusters: ")
for cluster in kmeans1.clusters:
    print(cluster)
    temp_x = []
    temp_y = []
    for elem in cluster:
        temp_x.append(elem[0])
        temp_y.append(elem[1])
    X.append(temp_x)
    Y.append(temp_y)

plt.scatter(X[0], Y[0], color='red')
plt.scatter(X[1], Y[1], color='blue')
plt.scatter(X[2], Y[2], color='green')
plt.title("t = 1")
plt.show()