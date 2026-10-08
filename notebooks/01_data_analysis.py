from pandas import *

df=read_csv('../dataset/heart_disease_uci.csv')
print("=" * 50)
print("first five rows ")
print("=" * 50)
print(df.head())

print("\n"+"="*50)
print("dataset shape")
print("=" *50)
print(df.shape)

print("\n"+"="*50)
print("columns names")
print("=" * 50)
print(df.columns.tolist())  

print("\n"+"="*50)
print("dataset info")
print("=" * 50)
print(df.info())

print("\n"+"="*50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe())

print("\n"+"="*50)
print("missing values ")
print("=" * 50)
print(df.isnull().sum())

print("\n"+"="*50)
print("duplicate values")
print("=" * 50)
print(df.duplicated().sum())

print("\n"+"="*50)
print("target distribution")
print("=" * 50)
print(df['num'].value_counts().sort_index())


