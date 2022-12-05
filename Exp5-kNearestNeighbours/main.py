import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import scipy.spatial
from collections import Counter

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
            votes = []
            for j in range(len(X_train)):
                dist = scipy.spatial.distance.euclidean(X_train[j] , X_test[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.k]
            for d, j in d:
                votes.append(Y_train[j])
            ans = Counter(votes).most_common(1)[0][0]
            final_output.append(ans)
            
        return final_output
    
    def score(self, X_test, Y_test):
        pred = self.predict(X_test)
        return (pred == Y_test).sum() / len(Y_test)

data = pd.read_csv("data.csv")
data = data.dropna()

X = data.iloc[:,1:-1]
X = X.to_numpy()
X = normalize(X, axis=0, norm='max')

Y = data.iloc[:,-1]
Y = Y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

classifier = KNN(30)
classifier.fit(X_train, Y_train)
print(classifier.score(X_test, Y_test))