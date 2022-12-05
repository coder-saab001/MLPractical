import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')

# Taking out the instances from data
instances = np.array(data.iloc[:, :-1])
print("\nInstances are:\n", instances)

# Taking out the target from data
target = np.array(data.iloc[:, -1])
print("\nTarget Values are: ", target)

# Learn Function
def learn(concepts, target):
    # Initialization
    specific_hypothesis = concepts[0].copy()
    print("\n---------------------Initialization----------------------")
    print("Specific Boundary: ", specific_hypothesis)
    general_hypothesis = [["?" for i in range(len(specific_hypothesis))] for i in range(len(specific_hypothesis))]
    print("Generic Boundary: ", general_hypothesis)

    for i, val in enumerate(concepts):
        print("\n----------------------Instance ", i, "-----------------------")
        print("Instance is ", val)
        # positive example
        if target[i] == "yes":
            print("Instance is Positive ")
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                    general_hypothesis[x][x] = '?'
        # negative example
        if target[i] == "no":
            print("Instance is Negative ")
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    general_hypothesis[x][x] = specific_hypothesis[x]
                else:
                    general_hypothesis[x][x] = '?'

        print("Specific Bundary: ", specific_hypothesis)
        print("Generic : ", general_hypothesis)


    # Removing the empty enteries in general hyposthesis
    indices = [i for i, val in enumerate(general_hypothesis) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_hypothesis.remove(['?', '?', '?', '?', '?', '?'])

    return specific_hypothesis, general_hypothesis


specific_final, general_final = learn(instances, target)

# Printing final specific and general algorithm
print("\nFinal Specific hypothesis is: ", specific_final, sep="\n")
print("Final General hypothesis is: ", general_final, sep="\n")