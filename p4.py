import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()

df = pd.DataFrame(boston.data)

df.columns = boston.feature_names

df.head()

df['Price'] = boston.target

x = boston.data
y = boston.target

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(xtrain, ytrain)

ypred = reg.predict(xtest)

import seaborn as sn
sn.scatterplot(ypred, ytest)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest, ypred)
mse