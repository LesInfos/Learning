#K-Nearest Neighbors from scratch

import numpy as np
from math import sqrt
#import matplotlib.pyplot as plt
import warnings
#from matplotlib import style
from collections import Counter
import pandas as pd
import random

#style.use('dark_background')

dataset = {'w' : [[1,2],[2,3], [3,1]], 'r': [[5,6],[7,7],[6,8]] }
new_feats = [2,-100]


#[[plt.scatter(p[0],p[1],s = 100, color = i) for p in dataset[i]]for i in dataset]
#plt.scatter(new_feats[0], new_feats[1])
#plt.show()


def k_near(data, predict, k = 3):
    if len(data) >= k:
        warnings.warn("K should be greater than total voting groups")

    dists = []
    for group in data:
        for features in data[group]:
            dist = np.linalg.norm(np.array(features)-np.array(predict))
            dists.append([dist,group])

    votes = [i[1] for i in sorted(dists)[:k]]
    result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][0] / k
    return result , confidence * 100


#results = k_near(dataset, new_feats, k = 3)
#print (results)

df = pd.read_csv('breast-cancer-wisconsin(1).data')
df.replace('?',-99999, inplace=True)
df.drop(['id'],1, inplace=True)
df = df.astype(float).values.tolist()
random.shuffle(df)

test_size = .2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = df[:-int(test_size*len(df))]
test_data = df[-int(test_size*len(df)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_near(train_set, data, k = 5)
        if group == vote:
            correct +=1
        total += 1

print ('Accuracy',correct/total)