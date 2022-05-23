import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')

df.head(10)

df.describe()

df.shape

df.info()

df.size

plt.figure(figsize = (10,10))
sn.boxplot(x = 'sex', y = 'age', hue = df['survived'], data = df)