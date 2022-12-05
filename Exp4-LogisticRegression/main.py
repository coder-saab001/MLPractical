import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Reading dataset
df = pd.read_csv('bank.csv', sep = ';', header=0)
df.head()

# Dropping NaN columns
df = df.dropna()
print (df.shape)

# Dropping unnecessary datasets
print(list(df.columns))
df.drop(df.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)
print(df.head())

# creating one hot encoding of the categorical columns.
data = pd.get_dummies(df, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
print(df.head())

# Dropping the columns ending with unknown
data.drop(data.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)
print(data.columns)

# Now, data is ready for model building
print(data.head())
X = data.iloc[:,1:]
print(X.head())

Y = data.iloc[:,0]
print(Y.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

classifier = LogisticRegression(solver='lbfgs',random_state=0)
classifier.fit(X_train, Y_train)

predicted_y = classifier.predict(X_test)
print(predicted_y)

for x in range(len(predicted_y)):
    if (predicted_y[x] == 1):
        print(x, end="\t")

print('\nAccuracy: {:.2f}'.format(classifier.score(X_test, Y_test)))