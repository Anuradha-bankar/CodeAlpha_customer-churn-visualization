import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import os
print(os.getcwd())

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", encoding="latin1")
print(df.head())

#unwanted coloums remove
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print(df.head())
print(df.shape)

df.info()
df.describe()
df.duplicated().sum()
df.isnull().sum()
df
df.groupby("customerID")["gender"].sum().reset_index()