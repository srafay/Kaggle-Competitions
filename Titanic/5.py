# Data Preprocessing

# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5,6,7]].values
y = dataset.iloc[:, 1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 2:5])
X[:, 2:5] = imputer.transform(X[:, 2:5])
#imputer = imputer.fit(X[:, 6:7])
#X[:, 6:7] = imputer.transform(X[:, 6:7])

#Encoding the sex column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)



# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((891, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()



testset = pd.read_csv('test.csv')
Xtest = testset.iloc[:, [1,3,4,5,6]].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(Xtest[:, 2:5])
Xtest[:, 2:5] = imputer.transform(Xtest[:, 2:5])

labelencoder_Xtest = LabelEncoder()
Xtest[:, 1] = labelencoder_Xtest.fit_transform(Xtest[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])
Xtest = onehotencoder.fit_transform(Xtest).toarray()

#Moving the columns to appropriate positions
Xtest[:, 0], Xtest[:, 1] = Xtest[:, 1], Xtest[:, 0].copy()

y_pred = classifier.predict(Xtest)

import numpy
numpy.savetxt("foo.csv", y_pred, delimiter=",", fmt='%.1d')