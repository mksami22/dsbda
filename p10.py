import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')

df.head(10)

df.describe()

df.info()

df.size

df.shape

df.hist()
plt.show()

df.boxplot()
plt.show()