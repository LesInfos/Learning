#   SKT Learn Linear Regression Demo

import pandas as pd
import quandl as q
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = q.get('WIKI/GOOGL')


df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] *100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] *100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]


forecast = 'Adj. Close'
df.fillna(-999999, inplace = True)

forecast_out = int(math.ceil(.001 * len(df)))

df['label'] = df[forecast].shift(-forecast_out)

#print(df.head())

X = np.array(df.drop(['label'],1))

y = np.array(df['label'])

X = preprocessing.scale(X)

y = np.array(df['label'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

classifier1 = svm.SVR() #LinearRegression() #use any ML model here
classifier1.fit(X_train, y_train)
acc = classifier1.score(X_test,y_test)

print(acc)
