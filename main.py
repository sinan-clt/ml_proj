import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1

data = pd.read_csv('data.csv')

print(data.head())

print(data.isnull().sum())

corr_matrix = data.corr()
plt.matshow(corr_matrix)
plt.xticks(range(len(data.columns)), data.columns, rotation=90)
plt.yticks(range(len(data.columns)), data.columns)
plt.colorbar()
plt.show()

# 2:

data = data.drop('Course_Instructor', axis=1)

data['Class_attribute'] = data['Class_attribute'].map({'low': 0, 'medium': 1, 'high': 2})

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': [2, 5, 10, None]}

rfc = RandomForestClassifier(random_state=42)
rfc_cv = GridSearchCV(rfc, param_grid, cv=5)

rfc_cv.fit(X_train, y_train)

print("Best Parameters: ", rfc_cv.best_params_)
print("Best Score: ", rfc_cv.best_score_)

# 5
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = rfc_cv.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1)

import pickle

filename = 'finalized_model.sav'
pickle.dump(rfc_cv, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

y_pred_loaded = loaded_model.predict(X_test)

print(np.array_equal(y_pred, y_pred_loaded))

