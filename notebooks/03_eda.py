from pandas import *
from matplotlib import pyplot as plt
from seaborn import *

df=read_csv('../dataset/heart_disease_uci.csv')

df["num"]=df["num"].apply(lambda x: 0 if x == 0 else 1)

df=df.drop("id",axis=1)

plt.figure(figsize=(10,6))
countplot(x="num",data=df)
plt.title("heart disease distribution")
plt.xlabel("heart disease")
plt.ylabel("num of patients")
plt.show()

plt.figure(figsize=(10,6))
histplot(data=df,x="age",kde=True)
plt.title("age distribution")
plt.show()

plt.figure(figsize=(10,6))
boxplot(x="num",y="age",data=df)
plt.title("age vs heart disease")
plt.xlabel("heart disease")
plt.ylabel("age")
plt.show()

plt.figure(figsize=(10,6))
countplot(x="sex",hue="num",data=df)
plt.title("sex vs heart disease")
plt.xlabel("sex")
plt.ylabel("num of patients")
plt.show()

plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

heatmap(numeric_df.corr(),annot=True)
plt.title("correlation heatmap")
plt.show()

