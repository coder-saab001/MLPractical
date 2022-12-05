import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.svm import SVC 

# Read in the csv
df = pd.read_csv('data.csv', encoding='utf-8')
df['rating_difference'] = df['white_rating'] - df['black_rating']
df['white_win'] = df['winner'].apply(lambda x : 1 if x == 'white' else 0)
df.iloc[: , [0,1,5,6,8,9,10,11,13,16,17]]
def fitting(X, y, C, gamma):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    model = SVC(kernel = 'linear', random_state=0)
    clf = model.fit(X_train, y_train)
    pred_labels_tr = model.predict(X_train)
    pred_labels_te = model.predict(X_test)

    print('----- Evaluation on Test Data -----')
    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)

    print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')
    print('----- Evaluation on Training Data -----')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')

    return X_train, X_test, y_train, y_test, clf

X=df[['rating_difference', 'turns']]
y=df['white_win'].values

X_train, X_test, y_train, y_test, clf = fitting(X, y, 1, 'scale')