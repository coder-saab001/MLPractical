import pandas as pd
import numpy as np
# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split 
import math
from sklearn.metrics import confusion_matrix

def find_distance(x, y):
    distance = 0
    for i in range(0, len(x)):
        distance += pow(x[i]-y[i], 2)
    return math.sqrt(distance)

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X_test):
        final_output = []
        for i in range(len(X_test)):
            d = []
            for j in range(len(X_train)):
                dist = find_distance(X_train[j], X_test[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.k]
            votes = {}
            max_votes = 0
            max_class = ""
            for dist,j in d:
                curr_class = Y_train[j]
                if curr_class in votes:
                    votes[curr_class] += 1
                else:
                    votes[curr_class] = 1
                
                if votes[curr_class] > max_votes:
                    max_votes = votes[curr_class]
                    max_class = curr_class
            final_output.append(max_class)
        return final_output
            

    def score(self, X_test, Y_test):
        predicted = self.predict(X_test)
        return (predicted == Y_test).sum() / len(Y_test)
    
data = pd.read_csv('data.csv')
data = data.dropna()

X = data.iloc[:,1:-1]
X = X.to_numpy()
# X = normalize(X, axis=0, norm='max')

Y = data.iloc[:, -1]
Y = Y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

classifier = KNN(30)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

pred = {"Actual":[], "Predicted":[]}
for i in range(len(Y_test)):
    pred["Actual"].append(Y_test[i])
    pred["Predicted"].append(Y_pred[i])
pred = pd.DataFrame.from_dict(pred)
print(pred)

cf_matrix = confusion_matrix(pred["Actual"], pred["Predicted"])
print(cf_matrix)