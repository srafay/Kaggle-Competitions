# Data Preprocessing

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5,6,7,9]].values
y = dataset.iloc[:, 1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
imputer = imputer.fit(X[:, 5:6])
X[:, 5:6] = imputer.transform(X[:, 5:6])

#Encoding the sex column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Moving the columns to appropriate positions
X[:, 0], X[:, 1] = X[:, 1], X[:, 0].copy()


# Kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X, y)


testset = pd.read_csv('test.csv')
Xtest = testset.iloc[:, [1,3,4,5,6,8]].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(Xtest[:, 2:3])
Xtest[:, 2:3] = imputer.transform(Xtest[:, 2:3])
imputer = imputer.fit(Xtest[:, 5:6])
Xtest[:, 5:6] = imputer.transform(Xtest[:, 5:6])

labelencoder_Xtest = LabelEncoder()
Xtest[:, 1] = labelencoder_Xtest.fit_transform(Xtest[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])
Xtest = onehotencoder.fit_transform(Xtest).toarray()

#Moving the columns to appropriate positions
Xtest[:, 0], Xtest[:, 1] = Xtest[:, 1], Xtest[:, 0].copy()

y_pred = classifier.predict(Xtest)

import numpy
numpy.savetxt("foo.csv", y_pred, delimiter=",", fmt='%.1d')