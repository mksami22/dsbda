import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/DeBugg14/te-sppu-lab/main/Data-Science/Prac2/prac2.csv')

df.head()

df.shape

df.size

df.info()

df.describe()

df.isnull().sum()

coln = df.columns
miss = []
for i in coln:
    t = df[i].isnull().sum()
    if t != 0:
        miss.append(i)
miss

pd.options.mode.chained_assignment = None
for j in miss:
    q=df[j].dtypes
    if (q=='int64' or q=='float64') :
        f=df[j]
        for k in range(df.shape[0]):
            if (f[k]<0 or f[k]>100) :
                f[k]=(np.nan)
    else:
        continue

for j in miss:
    q=df[j].dtypes
    if (q=='int64' or q=='float64') :
        df[j].fillna((df[j].mean()),inplace=True)
    else:
        df.fillna(method='bfill')
df.head(10)

df['Total Marks']=df['Phy_marks']+df['Che_marks']+df['EM1_marks']+df['PPS_marks']+df['SME_marks']
df['Percentage']=df['Total Marks']/5

df['Attendance'].plot(kind = 'box')

Q1 = df['Attendance'].quantile(0.25)
Q3 = df['Attendance'].quantile(0.75)
IQR = Q3 - Q1
Lower_limit = Q1 - 1.5*IQR
Upper_limit = Q3 + 1.5*IQR

df[(df['Attendance']<Lower_limit)|(df['Attendance']>Upper_limit)]