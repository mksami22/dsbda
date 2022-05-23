import pandas as pd
import numpy as np
import seaborn as sn

df = pd.read_csv('5.csv')

df.head()

x = df.iloc[:, [2, 3]].values
y = df.iloc[:, 4].values

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

reg.fit(xtrain, ytrain)

ypred = reg.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)
cm