# Data Preprocessing Template

# Importing the libraries
#---------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #the best library to import data to explore
#---------------------------------------------------------------------------------------------------------

# Importing the dataset
#---------------------------------------------------------------------------------------------------------

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
#---------------------------------------------------------------------------------------------------------

#Taking care of missing Data
#---------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3]) 
X[:,1:3] = imputer.transform(X[:,1:3])
#this implies that it will take column 1 and column 2, since Python excludes the column 3.
#imputer.fit take X columns 1 and 2 and uses the strategy described above for those missing values.
#then we have to place them back on X array with imputer.transform
#--------------------------------------------------------------------------------------------------------- 

#Encoding Categorical Data
#---------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() 
X[:,0] = labelencoder_X.fit_transform(X[:,0]) 
#This encodes the labels on the column 0 (Country) of array X and place them again in X column 0.
# where: 0 = France, 1 = Germany,  2 = Spain
onehotencoder = OneHotEncoder(categorical_features = [0]) 
#this categorical_features = [0] specifies where the categorical data is located
X = onehotencoder.fit_transform(X).toarray()
#What the OneHotEncoder does is that it converts all the labels of the column 0 into "as many" columns,
#and then places a 1 in the corresponding column when that label is presented in X.
#Example:   France equals  1 0 0
#           Germany equals 0 1 0
#           Spain equals   0 0 1
# in this case there are 3 columns because there are 3 categories inside column 0 (values 0,1,2).
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y) 
#this makes variable y being 1s and 0s instead of YES and NO
#---------------------------------------------------------------------------------------------------------

# Splitting the dataset into the Training set and Test set
#---------------------------------------------------------------------------------------------------------

#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
##---------------------------------------------------------------------------------------------------------
#

## Feature Scaling
#"""from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)"""