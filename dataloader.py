from sklearn.model_selection import train_test_split
from typing import Dict
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nlp
import numpy as np
import pandas as pd


# Read data
def load_data(filepath):
    wine_data=pd.read_csv(filepath,index_col=0)
    #display(wine_data.head())

    # subset
    df=wine_data[['description','taster_name']]
    return df

def feature_engineering(df):
    #print(df.isna().sum())
    print('Value_counts',df['taster_name'].value_counts())
    #print(len(df['taster_name']))
    df=df[~df['taster_name'].isna()]
    print('Check null values',df.isna().sum())
    #df['taster_name'].value_counts().plot(kind='bar')
    # label encoding for names
    le=LabelEncoder()
    df['taster_label']=le.fit_transform(df['taster_name'])
    df.drop('taster_name',axis=1)
    print('Category count',len(df['taster_label'].unique()))
    return df

def data_split(df):
    # train test val split
    training_data, testing_data= train_test_split(df,test_size=0.3,random_state=25)
    val_data, testing_data=train_test_split(testing_data,test_size=0.5,random_state=25)

    train=nlp.Dataset.from_pandas(training_data,split='train')
    test=nlp.Dataset.from_pandas(testing_data,split='test')
    val=nlp.Dataset.from_pandas(val_data,split='val')
    return train,test,val,testing_data

def start_preprocessing(filepath):
    df=load_data(filepath)
    df=feature_engineering(df)
    train,test,val,testing_data=data_split(df)
    return train,test,val,testing_data







    