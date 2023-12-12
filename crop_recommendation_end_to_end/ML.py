# -*- coding: utf-8 -*-
#Importing the libraries 

#import numpy as np
#import matplotlib.pyplot as plt
'''import pandas as pd



#Importing the dataset 

dataset = pd.read_csv('Crop_recommendation.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



#splitting the dataset into training set and test set 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



#Feature Scaling 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Training the naive bayes model on the training set 

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



#API call for weather Data 


import requests 
city_name='New Delhi'
weather_data = requests.get('http://api.weatherstack.com/current?access_key=282e3892595fa9b15764697842c943c2&query={}'.format(city_name))
observation_time = ''
temperature = ''
weather_description = ''
humidity = ''
observation_time = weather_data.json()['current']['observation_time']
temperature = weather_data.json()['current']['temperature']
weather_description = weather_data.json()['current']['weather_descriptions']
humidity = weather_data.json()['current']['humidity']

#Prediciting a new result 

print(classifier.predict(sc.transform([[91,43,44,21.87974,83.00027,6.5111,203.9355]])))


#Making a pickle file of the Classsifier 


import pickle 
filename = 'classifier.pkl'
pickle.dump(classifier,open(filename,'wb'))

filename2 = 'standard_scalar.pkl'
pickle.dump(sc,open(filename2,'wb')) '''


#Deep Learning 



#Importing the libraries

import numpy as np
import pandas as pd
import tensorflow as tf


#Importing the dataset

dataset = pd.read_csv('Crop_recommendation.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


y = y.reshape(-1, 1)

#one hot-encoding the dependent variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [0])], remainder='passthrough',)
y= np.array(ct.fit_transform(y))






#Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, train_size=0.80, random_state = 1)



#Feature Scaling


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Initializing the ANN


ann = tf.keras.models.Sequential()



#Adding layers to the ANN


ann.add(tf.keras.layers.Dense(units=7, activation='relu'))
ann.add(tf.keras.layers.Dense(units=7, activation='relu'))
ann.add(tf.keras.layers.Dense(units=22, activation='softmax')) #Output layer 


#Compiling the ANN


ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#Training the ANN on the Training set

ann.fit(X_train, y_train, batch_size = 128, epochs = 100)


#Predicting a result 

pred = ann.predict(sc.transform([[91,43,44,21.87974,83.00027,6.5111,203.9355]]))
pred = ct.named_transformers_['encoder'].inverse_transform(pred)
print(pred)



#Making a pickle file of the Classsifier 


import pickle 
filename = 'ann.pkl'
pickle.dump(ann,open(filename,'wb'))
from tensorflow.keras.models import load_model
ann.save('ann.h5')

filename2 = 'standard_scalar.pkl'
pickle.dump(sc,open(filename2,'wb'))

filename3 = 'column_transformer.pkl'
pickle.dump(ct,open(filename3,'wb'))
















































