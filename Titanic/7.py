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

#Removing Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#accuracies.mean()
#accuracies.std()


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

Xtest = Xtest[:, 1:]

y_pred = classifier.predict(Xtest)
y_pred = (y_pred > 0.5)

np.savetxt("foo.csv", y_pred, delimiter=",", fmt='%.1d')