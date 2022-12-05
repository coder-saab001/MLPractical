import numpy as np
import pandas as pd

csvreader = pd.read_csv("data.csv")
data = np.array(csvreader)[:,:-1]
target = np.array(csvreader)[:,-1]

def train(dataset, target) :
    n = len(dataset)
    for i in range(n):
        if target[i] == "yes":
            ans = dataset[i].copy()
            break
    
    for i in range(n):
        if target[i] == "yes":
            for j in range(len(ans)):
                if dataset[i][j] != ans[j]:
                    ans[j] = '?'
    return ans

ans = train(data, target)
print(ans)