import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", encoding="latin1")
print(df.head())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
print(df.dtypes)

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

#feature distribution
plt.figure(figsize=(12, 8))
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()


#Heat map 
numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()
corr_long = corr_matrix.reset_index().melt(id_vars='index')
corr_long.columns = ['Feature1', 'Feature2', 'Correlation']
fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    zmin=-1,
    zmax=1,
    aspect='auto',
    title='Correlation Heatmap of Numeric Features (Telco Customer Churn)'
)

fig.show()


#Bubble Scatter: Monthly vs Total Charges
fig = px.scatter(
    df,
    x='MonthlyCharges',
    y='TotalCharges',
    size='tenure',
    color='Contract',
    hover_data=[
        'Churn',
        'InternetService',
        'PaymentMethod'
    ],
    title='Monthly Charges vs Total Charges by Contract Type',
    size_max=60
)

fig.show()

#Bubble Scatter: Total Charges vs Tenure
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

fig = px.scatter(
    df,
    x='TotalCharges',          
    y='tenure',                
    size='MonthlyCharges',     
    color='Churn',            
    hover_data=[
        'Contract', 
        'InternetService', 
        'PaymentMethod',
        'SeniorCitizen',
        'gender'
    ],
    title='Total Charges vs Tenure with Monthly Charges Bubble Size',
    size_max=60
)

fig.show()

#Tree Map 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

fig = px.treemap(
    df,
    path=['Contract'],     
    values='TotalCharges',
    title='Treemap of Total Charges by Contract Type'
)

fig.show()


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['Churn'] = df['Churn'].astype(str).str.strip().str.capitalize()  

df['Churn_binary'] = df['Churn'].map({'Yes':1, 'No':0})

treemap_df = df.groupby(['Contract', 'InternetService', 'PaymentMethod'], as_index=False).agg(
    TotalCharges_sum=('TotalCharges','sum'),
    Churn_rate=('Churn_binary','mean'),  
    Count=('Churn_binary','count')
)

fig = px.treemap(
    treemap_df,
    path=['Contract', 'InternetService', 'PaymentMethod'],
    values='TotalCharges_sum',
    color='Churn_rate',
    color_continuous_scale='RdYlGn_r',  
    title='Multi-level Treemap: Contract → InternetService → PaymentMethod with Churn Rate',
    hover_data=['Count','Churn_rate']
)

fig.show()

df['Churn'].value_counts()

#Line Plot: Total Charges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df_plot = df.dropna(subset=['tenure','TotalCharges','Contract','Churn'])

df_plot_sorted = df_plot.sort_values('tenure')

df_smoothed = df_plot_sorted.groupby(['Contract','Churn']).apply(
    lambda x: x.assign(
        TotalCharges_Smooth = x['TotalCharges'].rolling(window=5, min_periods=1).mean()
    )
).reset_index()

print(type(df_smoothed))
print(df_smoothed.columns)

fig = px.line(
    df_smoothed,
    x='tenure',
    y='TotalCharges_Smooth',
    color='Churn',
    line_dash='Contract',
    markers=True,
    title='Smoothed Total Charges by Tenure, Contract Type and Churn'
)

fig.show()

#pie Chart: Top 10 Contracts
top_contracts = df['Contract'].value_counts().nlargest(10).reset_index()
top_contracts.columns = ['Contract', 'Count']

fig = px.pie(
    top_contracts,
    names='Contract',
    values='Count',
    title='Top 10 Contract Types'
)

pull = [0.1 if i == 0 else 0 for i in range(len(top_contracts))]
fig.update_traces(textinfo='percent+label', pull=pull)

fig.update_layout(legend_title_text='Contract Type')

fig.show()


#Pie Chart: Payment Methods
top_payment = df['PaymentMethod'].value_counts().nlargest(10).reset_index()
top_payment.columns = ['PaymentMethod', 'Count']

fig = px.pie(
    top_payment,
    names='PaymentMethod',
    values='Count',
    title='Top 10 Payment Methods'
)

pull = [0.1 if i == 0 else 0 for i in range(len(top_payment))]
fig.update_traces(textinfo='percent+label', pull=pull)
fig.update_layout(legend_title_text='Payment Method')
fig.show()
