import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')
instances = np.array(data.iloc[:,:-1])
targets = np.array(data.iloc[:,-1])

def train(concepts, targets):
    n = len(concepts[0])
    most_specific_hypothesis = ['$' for i in range(n)]
    most_general_hypothesis = ['?' for i in range(n)]
    specific_hypothesis = most_specific_hypothesis.copy()
    general_hypothesis = [most_general_hypothesis.copy() for i in range(n)]
    print("\n------------------Initialization------------------")
    print("Specific Boundary: ", specific_hypothesis)
    print("General Boundary: ", general_hypothesis)
    for i in range(len(concepts)):
        print("\n------------------Instance ", i, " ------------------")
        print("Instance is: ", concepts[i])
        if targets[i] == 'yes':
            print("Instance is Positive")
            if np.array_equal(specific_hypothesis, most_specific_hypothesis):
                specific_hypothesis = concepts[i]
                pass
            for j in range(n):
                if specific_hypothesis[j] != concepts[i][j]:
                    specific_hypothesis[j] = '?'
                    general_hypothesis[j][j] = '?'
        else:
            print("Instance is Negative")
            for j in range(n):
                if specific_hypothesis[j] != concepts[i][j]:
                    general_hypothesis[j][j] = specific_hypothesis[j]
                else:
                    general_hypothesis[j][j] = '?'
        print("Specific Boundary: ", specific_hypothesis)
        print("General Boundary: ", general_hypothesis)
    
    count_empty = 0
    for i in range(n):
        count_empty += np.array_equal(general_hypothesis[i], most_general_hypothesis)
    for i in range(count_empty):
        general_hypothesis.remove(most_general_hypothesis)
    
    return specific_hypothesis, general_hypothesis

specific_hypothesis, general_hypothesis = train(instances, targets)
print("\n------------------Final Output------------------")
print("Final Specific Hypothesis: ", specific_hypothesis)
print("Final General Hypothesis: ", general_hypothesis)