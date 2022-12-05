import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')
instances = np.array(data)[:,:-1]
targets = np.array(data)[:,-1]

n = len(instances[0])
specific_boundary = instances[0]
general_boundary = [['?' for i in range(n)] for i in range(n)]
print("\n-----------------Initialization----------------")
print("Specific Boundary:\n", specific_boundary)
print("General Boundary:\n", general_boundary)
for i in range(len(instances)):
    print("\n-----------------Instance", i, "----------------")
    if targets[i] == 'yes':
        print("Instance is positive")
        for j in range(n):
            if specific_boundary[j] != instances[i][j]:
                specific_boundary[j] = '?'
                general_boundary[j][j] = '?'

    else:
        print("Instance is Negative")
        for j in range(n):
            if specific_boundary[j] != instances[i][j]:
                general_boundary[j][j] = specific_boundary[j]
            else:
                general_boundary[j][j] = '?'
    print("Specific Boundary:\n", specific_boundary)
    print("General Boundary:\n", general_boundary)

count_empty = 0
for i in range(n):
    count_empty += np.array_equal(general_boundary[i],['?' for i in range(n)])
for i in range(count_empty):
    general_boundary.remove(['?' for i in range(n)]) 

print("\n-----------------Finalization----------------")
print("Specific Boundary:\n", specific_boundary)
print("General Boundary:\n", general_boundary)