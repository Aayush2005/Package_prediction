##Feature analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('Travel.csv')
print(df.isnull().sum())
print(df.info())

print(df['Gender'].value_counts())
print(df['MaritalStatus'].value_counts())
print(df['TypeofContact'].value_counts())
print(df['Occupation'].value_counts())
print(df['ProductPitched'].value_counts())
print(df['Designation'].value_counts())

df['Gender'] = df['Gender'].replace('Fe Male','Female')
df['MaritalStatus'] = df['MaritalStatus'].replace('Single','Unmarried')

print(df['Gender'].value_counts())
print(df['MaritalStatus'].value_counts())

#Checking for NaN values

feature_with_nan = [features for features in df.columns if df[features].isnull().sum()>=1]
for feature in feature_with_nan:
    print(feature,np.round(df[feature].isnull().mean()*100,5),'% missing values')

##Handling Missing values that is Filling Numerical features with Median and Categorical with Mode

df.Age.fillna(df.Age.median(),inplace=True)
df.TypeofContact.fillna(df.TypeofContact.mode()[0],inplace=True)
df.DurationOfPitch.fillna(df.DurationOfPitch.median(),inplace=True)
df.NumberOfFollowups.fillna(df.NumberOfFollowups.mode()[0],inplace = True) #Discrete Feature
df.PreferredPropertyStar.fillna(df.PreferredPropertyStar.mode()[0],inplace = True)
df.NumberOfTrips.fillna(df.NumberOfTrips.median(),inplace = True)
df.NumberOfChildrenVisiting.fillna(df.NumberOfChildrenVisiting.mode()[0],inplace = True)
df.MonthlyIncome.fillna(df.MonthlyIncome.median(),inplace=True)

print(df.isnull().sum())

df.drop('CustomerID',inplace=True,axis = 1)

##Since number of Children visiting and number of people Visiting ar e to differeent Columns, We can combine them
df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df.drop(columns = ['NumberOfChildrenVisiting','NumberOfPersonVisiting'],inplace = True, axis = 1)
print(df.info())

df.to_csv("Travel_Cleaned.csv",index=False)