import pandas as pd
import numpy as np
import seaborn as sn

df = pd.read_csv('iris.csv')

df.head()

x = df.iloc[:, :4].values
y = df['variety'].values

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

from sklearn.naive_bayes import GaussianNB
reg = GaussianNB()

reg.fit(xtrain, ytrain)

ypred = reg.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)
cm