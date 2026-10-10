from pandas import *

df=read_csv('../dataset/heart_disease_uci.csv')

print("original dataset shape")
print(df.shape)

df=df.drop("id",axis=1)


df["num"] = df["num"].apply(
    lambda x: 0 if x == 0 else 1
)

print("duplicate values")
print(df.duplicated().sum())

df=df.drop_duplicates()
print(df.shape)

print(" targer distribution")
print(df["num"].value_counts())

print("clean data")
print(df.head())
