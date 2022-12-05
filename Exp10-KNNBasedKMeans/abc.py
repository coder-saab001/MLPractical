import numpy as np
import pandas as pd
from math import sqrt, inf

data = pd.read_csv('kmeans.csv')

X = data['X'].values
Y = data['Y'].values
n = len(X)
pairs = []
for i in range(n):
    pairs.append((X[i], Y[i]))

T = 4
clusters = []
clusters.append([])
clusters[0].append(pairs[0])

def euc_dist(A, B):
    xa, ya = A[0], A[1]
    xb, yb = B[0], B[1]
    return sqrt((xa-xb)**2 + (ya-yb)**2)

num_cluster = 0
for i in range(1, n):
    temp_clus = clusters[0]
    min_d = inf
    found = False
    for cluster in clusters:
        for pt in cluster:
            dist = euc_dist(pt, pairs[i])
            if dist >= T:
                continue
            else:
                if dist < min_d:
                    found = True
                    min_d = dist
                    temp_clus = cluster
    if not found:
        num_cluster += 1
        clusters.append([])
        clusters[num_cluster].append(pairs[i])
    else:
        temp_clus.append(pairs[i])

for cluster in clusters:
    print(cluster)

