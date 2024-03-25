import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_data(file_path):
    df=pd.read_csv(file_path)

    encoder1=LabelEncoder()
    encoder2=LabelEncoder()
    encoder3=LabelEncoder()
    encoder4=LabelEncoder()


    df['isFradulent']=encoder1.fit_transform(df['isFradulent'])
    df['isHighRiskCountry']=encoder2.fit_transform(df['isHighRiskCountry'])
    df['isForeignTransaction']=encoder3.fit_transform(df['isForeignTransaction'])
    df['Is declined']=encoder4.fit_transform(df['Is declined'])

    x=df.drop(['Merchant_id','Transaction date','isFradulent'],axis=1)
    y=df['isFradulent']

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    return x_train , y_train , x_test  , y_test