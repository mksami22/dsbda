import pandas as pd
import numpy as np
import seaborn as sn 
import matplotlib.pyplot as plt

df = pd.read_csv('1.csv')

df.head()

df.tail()

df.describe()

df.info()

df.shape

df.isnull().sum()

df['stroke'].replace(np.nan, df['stroke'].mean(), inplace = True)

df['horsepower'].replace(np.nan, df['horsepower'].mean(), inplace = True)

df['peak-rpm'].replace(np.nan, df['peak-rpm'].mean(), inplace = True)

df.isnull().sum()

df['city-mpl'] = 235*df['city-L/100km']
df.drop(columns = 'city-L/100km')

df.head()

df['horsepower'] = df['horsepower']/df['horsepower'].max()

df.head()

df.columns

dummy = pd.get_dummies(df['aspiration'])

df = pd.concat([df, dummy], axis = 1)
df.drop(columns = 'aspiration')

df["horsepower"]=df["horsepower"].astype(float, copy=True)

bin = np.linspace(df['horsepower'].min(), df['horsepower'].max(), 4)

group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bin, labels=group_names, include_lowest=True )

df['horsepower-binned'].head()